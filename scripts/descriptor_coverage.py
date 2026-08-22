"""
Report rich-descriptor coverage and prototype collisions per held-out scale.

WHY THIS IS A SCRIPT AND NOT A HAND-COUNT
-----------------------------------------
The collision counts were first tallied by hand and reported as 30 collided classes at scale C. The
true figure is 26: four of the "collisions" were an underscore-normalisation bug in the matcher
(13 of 32 bank keys are multi-word and were unreachable against labels like `Powdery_Mildew`). A
number that appears in the paper as a finding about method must be regenerated from the code that
produces it, or the next matcher change silently invalidates it again.

    python scripts/descriptor_coverage.py            # print
    python scripts/descriptor_coverage.py --write    # also write docs/paper/descriptor_coverage.json
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import descriptors as D

DOC = C.REPO_ROOT / "docs" / "paper"


def held_labels(exp: str) -> list[str]:
    """Held-out labels for one configuration, taken from the released evaluation if present so the
    report always describes the same class list the accuracies were measured on."""
    p = DOC / f"zeroshot_eval_{exp}.json"
    if p.exists():
        cov = json.loads(p.read_text(encoding="utf-8")).get("coverage") or {}
        if cov:
            return sorted(cov)
    return []


def analyse(labels: list[str]) -> dict:
    cov: dict[str, str] = {}
    for lab in labels:
        D.text_for(lab, "rich", cov)
    groups: dict[str, list[str]] = defaultdict(list)
    no_match = []
    for lab, kw in cov.items():
        (no_match if kw == "(NO MATCH)" else groups[kw]).append(lab)
    unique = {k: v for k, v in groups.items() if len(v) == 1}
    collided = {k: v for k, v in groups.items() if len(v) > 1}
    n_collided = sum(len(v) for v in collided.values())
    return {
        "n_classes": len(labels),
        "no_match": len(no_match),
        "no_match_labels": sorted(no_match),
        "unique": len(unique),
        "collided_classes": n_collided,
        "collision_groups": {k: sorted(v) for k, v in
                             sorted(collided.items(), key=lambda kv: -len(kv[1]))},
        # each unmatched class still gets its own bare class-name string, so it is a distinct
        # prototype even though it carries no symptom text
        "distinct_prototypes": len(groups) + len(no_match),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    keys = [kw for kw, _ in D.RICH]
    multi = [k for k in keys if " " in k]
    out = {"bank_keys": len(keys), "bank_multiword_keys": len(multi), "scales": {}}

    print(f"rich bank: {len(keys)} keys ({len(multi)} multi-word)\n")
    print(f"{'scale':6s} {'classes':>8s} {'no-match':>9s} {'unique':>7s} {'collided':>9s} "
          f"{'distinct protos':>16s}")
    print("-" * 60)
    for e in "ABC":
        labels = held_labels(e)
        if not labels:
            print(f"{e:6s}  (no released evaluation to read the class list from)")
            continue
        a = analyse(labels)
        out["scales"][e] = a
        print(f"{e:6s} {a['n_classes']:8d} {a['no_match']:9d} {a['unique']:7d} "
              f"{a['collided_classes']:9d} {a['distinct_prototypes']:16d}")

    c = out["scales"].get("C")
    if c:
        print(f"\nscale C collision groups ({len(c['collision_groups'])} keywords, "
              f"{c['collided_classes']} classes):")
        for kw, labs in c["collision_groups"].items():
            print(f"   '{kw}' x{len(labs)}")
        print(f"\nscale C unmatched ({c['no_match']}): "
              f"{', '.join(l.split('|')[1] for l in c['no_match_labels'][:8])}"
              f"{' ...' if c['no_match'] > 8 else ''}")

    if args.write:
        p = DOC / "descriptor_coverage.json"
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[coverage] wrote {p}")


if __name__ == "__main__":
    main()
