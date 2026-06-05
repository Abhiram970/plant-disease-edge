"""
Phase A1 — Build the SAGE subset by STREAMING + FILTERING.

We DO NOT download the full ~114 GB SAGE parquet. We stream rows from the
HuggingFace `datasets` repo, keep only our 11 crops up to a per-class cap,
decode the image bytes, save JPEGs, and write a manifest.

WHERE TO RUN
------------
- PREFERRED: a Kaggle notebook (fast HF bandwidth; output -> a Kaggle Dataset).
- ALSO FINE: locally on the RTX 4060 box (no GPU needed for this step; it's I/O bound).
  Expect a few GB downloaded and ~6-8 GB written. Use --limit first to test cheaply.

USAGE
-----
  # 1) Inspect what crop strings SAGE actually uses (recommended first run):
  python scripts/build_sage_subset.py --probe --limit 5000

  # 2) Small test build (fast, proves the pipeline):
  python scripts/build_sage_subset.py --limit 20000 --per-class-cap 50

  # 3) Full subset build:
  python scripts/build_sage_subset.py

OUTPUT
------
  data/dataset_cleaned/<Crop>___<Disease>/<hash>.jpg
  data/manifest.csv   (columns: path, crop, disease, filename, split_role, sha1)
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Missing dependency: pip install datasets   (see requirements.txt)")
from PIL import Image
from tqdm import tqdm

# Allow running as `python scripts/build_sage_subset.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def to_jpeg_bytes(img_field) -> bytes | None:
    """SAGE 'image' column is decoded by `datasets` into a PIL image; re-encode to JPEG bytes.
    Returns None on unreadable images."""
    try:
        if isinstance(img_field, dict) and "bytes" in img_field and img_field["bytes"]:
            img = Image.open(io.BytesIO(img_field["bytes"]))
        else:
            img = img_field  # already a PIL.Image
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception:
        return None


def probe(limit: int) -> None:
    """Print the distribution of raw crop strings so we can verify alias matching."""
    print(f"[probe] streaming up to {limit} rows to inspect crop names...")
    ds = load_dataset(C.SAGE_HF_REPO, split="train", streaming=True)
    raw_counts, matched_counts = Counter(), Counter()
    for i, row in enumerate(ds):
        if i >= limit:
            break
        raw = row.get("crop")
        raw_counts[raw] += 1
        canon = C.canonical_crop(raw)
        if canon:
            matched_counts[canon] += 1
    print("\n=== raw crop strings seen (top 40) ===")
    for name, n in raw_counts.most_common(40):
        mark = "  <- KEEP" if C.canonical_crop(name) else ""
        print(f"  {n:6d}  {name!r}{mark}")
    print("\n=== mapped to our canonical crops (in this sample) ===")
    for crop in C.WANT_CROPS:
        print(f"  {matched_counts.get(crop, 0):6d}  {crop}")
    print("\nIf a wanted crop shows 0 here, check CROP_ALIASES in config.py.")


def build(args) -> None:
    C.ensure_dirs()
    cap = args.per_class_cap
    kept_per_class: dict[tuple[str, str], int] = defaultdict(int)
    seen_hashes: set[str] = set()
    rows_out: list[dict] = []
    stats = Counter()

    print(f"[build] streaming {C.SAGE_HF_REPO} (no full download). cap/class={cap}"
          + (f", row limit={args.limit}" if args.limit else ""))
    ds = load_dataset(C.SAGE_HF_REPO, split="train", streaming=True)

    pbar = tqdm(ds, desc="scanning rows", unit="row")
    for i, row in enumerate(pbar):
        if args.limit and i >= args.limit:
            break
        crop = C.canonical_crop(row.get("crop"))
        if crop is None:
            continue  # not one of our 11 crops
        disease = str(row.get("disease", "Unknown"))
        key = (crop, disease)
        if kept_per_class[key] >= cap:
            continue

        jpg = to_jpeg_bytes(row.get("image"))
        if jpg is None:
            stats["unreadable"] += 1
            continue
        h = sha1_bytes(jpg)
        if h in seen_hashes:
            stats["dup"] += 1
            continue
        seen_hashes.add(h)

        cls_dir = C.DATASET_DIR / f"{C.safe_name(crop)}___{C.safe_name(disease)}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{h[:16]}.jpg"
        (cls_dir / fname).write_bytes(jpg)

        kept_per_class[key] += 1
        stats["kept"] += 1
        rows_out.append({
            "path": str((cls_dir / fname).relative_to(C.DATA_ROOT)),
            "crop": crop,
            "disease": disease,
            "filename": fname,
            "split_role": "train_crop" if crop in C.TRAIN_CROPS else "heldout_crop",
            "sha1": h,
        })
        if stats["kept"] % 2000 == 0:
            pbar.set_postfix(kept=stats["kept"], dup=stats["dup"])

    # Drop tiny classes -> mark as OOD (move them out of train/heldout).
    class_counts = Counter((r["crop"], r["disease"]) for r in rows_out)
    dropped = {k for k, n in class_counts.items() if n < C.MIN_CLASS_IMAGES}
    for r in rows_out:
        if (r["crop"], r["disease"]) in dropped:
            r["split_role"] = "ood"

    # Write manifest.
    with open(C.MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "crop", "disease", "filename", "split_role", "sha1"])
        w.writeheader()
        w.writerows(rows_out)

    # Report.
    n_classes = len(class_counts)
    role_counts = Counter(r["split_role"] for r in rows_out)
    crop_counts = Counter(r["crop"] for r in rows_out)
    print("\n" + "=" * 60)
    print("SAGE SUBSET BUILD COMPLETE")
    print("=" * 60)
    print(f"  kept images   : {stats['kept']:,}")
    print(f"  duplicates    : {stats['dup']:,}   unreadable: {stats['unreadable']:,}")
    print(f"  classes       : {n_classes}   (dropped <{C.MIN_CLASS_IMAGES} imgs -> OOD: {len(dropped)})")
    print(f"  by role       : " + ", ".join(f"{k}={v:,}" for k, v in role_counts.items()))
    print("  by crop       :")
    for crop in C.WANT_CROPS:
        tag = "TRAIN" if crop in C.TRAIN_CROPS else "HELDOUT"
        print(f"     {crop:<10} [{tag:<7}] {crop_counts.get(crop, 0):,}")
    print(f"\n  manifest      : {C.MANIFEST_CSV}")
    print(f"  images        : {C.DATASET_DIR}")
    missing = [c for c in C.WANT_CROPS if crop_counts.get(c, 0) == 0]
    if missing:
        print(f"\n  WARNING: 0 images for {missing} — run with --probe and check CROP_ALIASES.")


def main():
    ap = argparse.ArgumentParser(description="Phase A1: stream-filter the SAGE subset.")
    ap.add_argument("--probe", action="store_true", help="just inspect crop strings, don't build")
    ap.add_argument("--limit", type=int, default=0, help="max rows to stream (0 = all; use for testing)")
    ap.add_argument("--per-class-cap", type=int, default=C.PER_CLASS_CAP)
    args = ap.parse_args()
    if args.probe:
        probe(args.limit or 5000)
    else:
        build(args)


if __name__ == "__main__":
    main()
