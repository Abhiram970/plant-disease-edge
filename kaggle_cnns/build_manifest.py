"""
Phase A1 (lightweight) — build manifest.csv from an existing folder-of-classes dataset.

The full streaming builder (build_sage_subset.py) was never needed once run_all.py fetched the
crops to disk. This scans DATASET_DIR (`<Crop>___<Disease>/<hash>.jpg`) and emits the manifest that
build_descriptors.py and the split/leakage tooling expect:

    path, crop, disease, filename, split_role

split_role is derived from config's crop lists:
    train_crop   crop in TRAIN_CROPS
    heldout_crop crop in HELDOUT_CROPS
    ood          anything else, or any class with < MIN images

USAGE
  # point at the existing run_all output, write manifest under PDE_DATA_ROOT:
  PDE_DATASET_DIR=/c/kaggle/working/exp_data PDE_DATA_ROOT=/c/kaggle/working \
      python scripts/build_manifest.py
  python scripts/build_manifest.py --min-images 15      # drop tiny classes to OOD
"""
from __future__ import annotations
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def split_role(crop: str, n: int, min_images: int) -> str:
    if n < min_images:
        return "ood"
    if crop in C.TRAIN_CROPS:
        return "train_crop"
    if crop in C.HELDOUT_CROPS:
        return "heldout_crop"
    return "ood"


def main():
    ap = argparse.ArgumentParser(description="Build manifest.csv from DATASET_DIR class folders.")
    ap.add_argument("--min-images", type=int, default=1,
                    help="classes with fewer images than this become split_role=ood")
    ap.add_argument("--out", default=str(C.MANIFEST_CSV))
    args = ap.parse_args()

    if not C.DATASET_DIR.exists():
        sys.exit(f"DATASET_DIR not found: {C.DATASET_DIR}\n"
                 f"Set PDE_DATASET_DIR to your image folder (e.g. C:\\kaggle\\working\\exp_data).")

    # first pass: count per (crop,disease)
    counts: Counter = Counter()
    folders = []
    for d in sorted(C.DATASET_DIR.iterdir()):
        if not d.is_dir() or "___" not in d.name:
            continue
        crop, disease = d.name.split("___", 1)
        jpgs = list(d.glob("*.jpg"))
        if not jpgs:
            continue
        counts[(crop, disease)] = len(jpgs)
        folders.append((crop, disease, jpgs))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    role_tally: Counter = Counter()
    crop_tally: Counter = Counter()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "crop", "disease", "filename", "split_role"])
        for crop, disease, jpgs in folders:
            role = split_role(crop, counts[(crop, disease)], args.min_images)
            role_tally[role] += len(jpgs)
            if role != "ood":
                crop_tally[crop] += len(jpgs)
            for jpg in jpgs:
                w.writerow([str(jpg), crop, disease, jpg.name, role])

    print(f"[manifest] wrote {out_path}")
    print(f"[manifest] images by role: " + ", ".join(f"{r}={n:,}" for r, n in role_tally.most_common()))
    print(f"[manifest] kept crops: " + ", ".join(f"{c}={n}" for c, n in sorted(crop_tally.items())))
    seen = sorted(c for c in crop_tally if c in C.TRAIN_CROPS)
    held = sorted(c for c in crop_tally if c in C.HELDOUT_CROPS)
    print(f"[manifest] train_crop={seen}")
    print(f"[manifest] heldout_crop={held}")


if __name__ == "__main__":
    main()
