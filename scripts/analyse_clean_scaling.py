#!/usr/bin/env python3
"""Decide the scaling claim from the cleaned and uncleaned evaluations.

The 2026-09-11 Kaggle run produced zeroshot_eval_{A,B,C}{,_clean}.json at all five
encoders. This script reads them and answers the three questions the manuscript's
scaling claim actually rests on, separately, because they have different answers:

  Q1  Does `grounded` rise monotonically as the label space grows?
  Q2  Does `rich` decay as the label space grows?
  Q3  Does `grounded` overtake `rich` at the largest scale?

The in-run verdict printed by kaggle/CELL_AUDIT.py tested Q1 only and reported
"withdraw the scaling claim", which is right about Q1 and misleading about the paper
as a whole: Q2 and Q3 both survive de-duplication, and Q3's margin widens.

    python scripts/analyse_clean_scaling.py                    # read from RESULTS_DIR
    python scripts/analyse_clean_scaling.py --dir path/to/json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STRATS = ("bare", "crude", "rich", "grounded")
CONFIGS = ("A", "B", "C")


def is_reference(model_key: str) -> bool:
    """The 92.9 M ceiling is reported but never averaged, as in make_tex_tables.py."""
    return "SigLIP" in model_key


def load(d: Path, cfg: str, clean: bool) -> dict | None:
    p = d / f"zeroshot_eval_{cfg}{'_clean' if clean else ''}.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text(encoding="utf-8"))
    models = {k: v for k, v in j["models"].items() if not is_reference(k)}
    if not models:
        return None
    return {
        "n_classes": j.get("n_classes"),
        "chance": j.get("chance"),
        "n_models": len(j["models"]),
        "mean": {s: 100.0 * sum(m[s]["acc"] for m in models.values()) / len(models)
                 for s in STRATS},
        # class-macro is present only if the run carried the by_class change
        "class_macro": {s: 100.0 * sum(m[s].get("class_macro_acc", 0.0) for m in models.values())
                        / len(models) for s in STRATS}
        if all("class_macro_acc" in m["grounded"] for m in models.values()) else None,
    }


def shape(v: list[float]) -> str:
    if v[0] < v[1] < v[2]:
        return "rises monotonically"
    if v[0] > v[1] > v[2]:
        return "falls monotonically"
    return "NOT monotone"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="directory holding the result JSONs")
    args = ap.parse_args()

    if args.dir:
        d = Path(args.dir)
    else:
        import config as C
        d = C.RESULTS_DIR
    print(f"reading from {d}\n")

    arms = {}
    for label, clean in (("uncleaned", False), ("cleaned", True)):
        rows = {c: load(d, c, clean) for c in CONFIGS}
        if any(r is None for r in rows.values()):
            missing = [c for c, r in rows.items() if r is None]
            print(f"!! {label}: missing or empty at {', '.join(missing)} -- skipping")
            continue
        short = [r["n_models"] for r in rows.values() if r["n_models"] < 5]
        if short:
            print(f"!! {label}: a file holds only {min(short)} encoders; expected 5. "
                  f"Re-run before trusting anything below.")
        arms[label] = rows

    for label, rows in arms.items():
        print(f"=== {label} — mean over the deployable tiers ===")
        head = "".join(f"{c} ({rows[c]['n_classes']} cls)".rjust(16) for c in CONFIGS)
        print(f"{'':10s}{head}      A->C")
        for s in STRATS:
            v = [rows[c]["mean"][s] for c in CONFIGS]
            print(f"  {s:8s}" + "".join(f"{x:16.2f}" for x in v)
                  + f"  {v[2] - v[0]:+7.2f}  {shape(v)}")
        if rows["C"]["class_macro"]:
            print("  -- class-macro --")
            for s in STRATS:
                v = [rows[c]["class_macro"][s] for c in CONFIGS]
                print(f"  {s:8s}" + "".join(f"{x:16.2f}" for x in v)
                      + f"  {v[2] - v[0]:+7.2f}  {shape(v)}")
        print()

    if len(arms) < 2:
        print("Need both arms to decide the claim.")
        return 1

    print("=" * 76)
    print("THE THREE QUESTIONS")
    print("=" * 76)
    for q, fn, keep in (
        ("Q1  grounded rises monotonically with the label space",
         lambda r: shape([r[c]["mean"]["grounded"] for c in CONFIGS]) == "rises monotonically",
         "abstract / contribution 2 / Section 4.3 / conclusion"),
        ("Q2  rich decays as the label space grows",
         lambda r: shape([r[c]["mean"]["rich"] for c in CONFIGS]) == "falls monotonically",
         "Section 4.3, Figure 1 caption, Table 2 note"),
        ("Q3  grounded overtakes rich at the largest scale",
         lambda r: r["C"]["mean"]["grounded"] > r["C"]["mean"]["rich"],
         "abstract crossover sentence"),
    ):
        res = {label: fn(rows) for label, rows in arms.items()}
        verdict = ("SURVIVES de-duplication" if all(res.values())
                   else "FALSIFIED by de-duplication" if res["uncleaned"] and not res["cleaned"]
                   else "was never supported" if not res["uncleaned"]
                   else "only holds after cleaning")
        print(f"\n{q}")
        for label, ok in res.items():
            print(f"    {label:10s}: {'yes' if ok else 'no'}")
        print(f"    -> {verdict}")
        print(f"    -> affects: {keep}")

    print("\n" + "=" * 76)
    print("MARGIN AT THE LARGEST SCALE (grounded - rich)")
    for label, rows in arms.items():
        m = rows["C"]["mean"]["grounded"] - rows["C"]["mean"]["rich"]
        print(f"  {label:10s}: {m:+.2f} points")
    print("\nCompare any of these against the 2.4-point seed-to-seed SD measured by the")
    print("control arm before describing it as a trend.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
