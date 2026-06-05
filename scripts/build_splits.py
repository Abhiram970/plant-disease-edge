"""
Phase A3 — Build train/val/test/heldout/ood splits from manifest.csv ONLY.

No external datasets. Splits come purely from the SAGE subset:
  - train_crop rows  -> stratified 80/10/10 train/val/test (per class)
  - heldout_crop rows -> ALL go to `heldout` (zero-shot eval; never trained)
  - ood rows          -> ALL go to `ood` (abstain-gate eval)

Then a content-hash LEAKAGE CHECK proves no image (by sha1) appears in two roles.

WHERE TO RUN: anywhere (pure CPU, instant). Needs data/manifest.csv from A1.

USAGE
-----
  python scripts/build_splits.py

OUTPUT
------
  data/splits/{train,val,test,heldout,ood}.csv   (same columns as manifest + 'split')
"""
from __future__ import annotations
import csv
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def load_manifest() -> list[dict]:
    if not C.MANIFEST_CSV.exists():
        sys.exit(f"manifest not found: {C.MANIFEST_CSV}\nRun build_sage_subset.py first (Phase A1).")
    with open(C.MANIFEST_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = load_manifest()
    rng = random.Random(C.RANDOM_SEED)
    tr_r, va_r, te_r = C.SPLIT_RATIOS

    # Group trained-crop rows by class for stratified splitting.
    by_class: dict[tuple[str, str], list[dict]] = defaultdict(list)
    heldout, ood = [], []
    for r in rows:
        role = r["split_role"]
        if role == "train_crop":
            by_class[(r["crop"], r["disease"])].append(r)
        elif role == "heldout_crop":
            heldout.append(r)
        elif role == "ood":
            ood.append(r)

    train, val, test = [], [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * tr_r)
        n_va = int(n * va_r)
        train += items[:n_tr]
        val += items[n_tr:n_tr + n_va]
        test += items[n_tr + n_va:]

    splits = {"train": train, "val": val, "test": test, "heldout": heldout, "ood": ood}

    # ---- LEAKAGE CHECK: no sha1 in more than one split ----
    sha_to_splits: dict[str, set[str]] = defaultdict(set)
    for name, items in splits.items():
        for r in items:
            sha_to_splits[r["sha1"]].add(name)
    leaks = {h: s for h, s in sha_to_splits.items() if len(s) > 1}
    if leaks:
        print(f"LEAKAGE DETECTED: {len(leaks)} images appear in multiple splits. Examples:")
        for h, s in list(leaks.items())[:5]:
            print(f"   {h[:12]} -> {sorted(s)}")
        sys.exit("Aborting — fix dedup in A1 before proceeding.")

    # ---- Write split CSVs ----
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["path", "crop", "disease", "filename", "split_role", "sha1", "split"]
    for name, items in splits.items():
        with open(C.SPLITS_DIR / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in items:
                w.writerow({**r, "split": name})

    # ---- Report ----
    print("=" * 60)
    print("SPLITS BUILT (leakage check PASSED — 0 shared images)")
    print("=" * 60)
    for name, items in splits.items():
        crops = sorted(set(r["crop"] for r in items))
        classes = len(set((r["crop"], r["disease"]) for r in items))
        print(f"  {name:<8} {len(items):>7,} imgs | {classes:>3} classes | crops: {crops}")
    print(f"\n  written to: {C.SPLITS_DIR}")
    print(f"  trained-crop classes: {len(by_class)}  (these are the student's label set)")


if __name__ == "__main__":
    main()
