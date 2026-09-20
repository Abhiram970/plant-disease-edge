#!/usr/bin/env python3
"""Turn hand-generated chat replies into the on-disk registries descriptors.py reads.

Reads manual_baselines/replies/<method>_<Crop>.{json,txt} and writes
descriptors_<method>/<seed>/<Crop>.json in the same shape build_descriptor_baselines.py
produces, so the scoring path cannot tell the two apart.

    python scripts/import_manual_baselines.py                 # import everything, seed 0
    python scripts/import_manual_baselines.py --seed 1        # a second draw
    python scripts/import_manual_baselines.py --check         # validate, write nothing

Tolerant of what a chat actually returns: ```json fences, prose before or after the object, and
disease keys written with spaces, hyphens or different capitalisation. It is NOT tolerant of a
class silently going missing -- every expected class is checked against the manifest and anything
absent is reported, because a missing class falls through to the keyword bank and would quietly
turn a DCLIP row into a part-bank row. That is the defect that invalidated the original `rich`
column, so it is a hard error here rather than a warning.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import config as C                      # noqa: E402
import build_descriptors as BD          # noqa: E402
import descriptors as D                 # noqa: E402

REPLIES = ROOT / "manual_baselines" / "replies"
FIELD = {"dclip": "descriptors", "cupl": "sentences"}
MIN_ITEMS = {"dclip": 4, "cupl": 4}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def extract_obj(raw: str) -> dict:
    """Pull the first balanced JSON object out of a chat reply."""
    s = re.sub(r"```(?:json)?", "", raw).strip()
    start = s.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    raise ValueError("JSON object is not balanced (reply truncated?)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = ap.parse_args()

    if not REPLIES.exists():
        sys.exit(f"no replies directory at {REPLIES}\n"
                 f"run scripts/make_manual_prompts.py first, then save each chat reply there "
                 f"as <method>_<Crop>.json")

    by_crop = BD.classes_from_manifest("heldout")
    problems: list[str] = []
    written = 0

    for method, field in FIELD.items():
        out_dir = C.REPO_ROOT / D.BASELINE_DIRS[method] / str(args.seed)
        for crop in sorted(by_crop):
            src = next((p for p in (REPLIES / f"{method}_{crop}.json",
                                    REPLIES / f"{method}_{crop}.txt") if p.exists()), None)
            if src is None:
                problems.append(f"{method}/{crop}: no reply file "
                                f"(expected {method}_{crop}.json)")
                continue
            try:
                obj = extract_obj(src.read_text(encoding="utf-8"))
            except Exception as e:
                problems.append(f"{method}/{crop}: unparseable -- {e}")
                continue

            lookup = {norm(k): v for k, v in obj.items()}
            recs, missing, thin = [], [], []
            for disease in sorted(by_crop[crop]):
                items = lookup.get(norm(disease))
                items = [str(x).strip() for x in items if str(x).strip()] if isinstance(items, list) else []
                if not items:
                    missing.append(disease)
                    continue
                if len(items) < MIN_ITEMS[method]:
                    thin.append(f"{disease}({len(items)})")
                recs.append({"crop": crop, "disease": disease,
                             "category": f"{disease} on {crop} leaf".replace("_", " "),
                             "method": method, "model": "manual-chat", "seed": args.seed,
                             "status": "filled", field: items})
            if missing:
                problems.append(f"{method}/{crop}: {len(missing)} class(es) absent from the "
                                f"reply -- {', '.join(missing)}")
            if thin:
                problems.append(f"{method}/{crop}: suspiciously few items for {', '.join(thin)}")
            if recs and not args.check:
                out_dir.mkdir(parents=True, exist_ok=True)
                p = out_dir / f"{C.safe_name(crop)}.json"
                p.write_text(json.dumps(recs, indent=2, ensure_ascii=False),
                             encoding="utf-8", newline="\n")
                written += 1
            n_items = sum(len(r[field]) for r in recs)
            print(f"  {method:5s} {crop:9s} {len(recs):2d}/{len(by_crop[crop])} classes, "
                  f"{n_items:4d} {field}")

    print()
    for p in problems:
        print(f"  [!] {p}")
    n_cls = sum(len(v) for v in by_crop.values())
    if args.check:
        print(f"\n--check: nothing written. {len(problems)} problem(s).")
    else:
        print(f"\nwrote {written} crop file(s) per method into "
              f"descriptors_{{dclip,cupl}}/{args.seed}/")
    if problems:
        print("Fix the flagged classes before scoring: a class with no entry falls through to "
              "the keyword bank, which would silently make these rows part-bank.")
        return 1
    print(f"All {n_cls} classes present for both methods. Score with:\n"
          f"  python scripts/evaluate.py --exp C --strategies bare80 dclip cupl --heavy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
