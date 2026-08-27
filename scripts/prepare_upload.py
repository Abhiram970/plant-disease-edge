"""
Turn the existing full-resolution local build into an upload-ready Kaggle dataset.

WHY THIS EXISTS INSTEAD OF A KAGGLE FETCH
-----------------------------------------
The published results were measured on the MAY 2026 release of SAGE (pinned in config.py as
SAGE_REVISION_MAY). That release is 114 GB in 13 shards of ~10.7 GB each, and pulling it through a
Kaggle session is the single riskiest step in the whole study -- it is what burned a 12 h session.
The same images are already on disk here at full resolution, so the sane move is to downscale them
once, locally, and upload ~2.5 GB instead of downloading 114 GB.

WHAT IT DOES
    * keeps only the 18 crops in config.WANT_CROPS (the local build also contains Cassava and
      Pumpkin from an earlier experiment; they are not part of this study)
    * caps each class at CAP_TRAIN_PER_CLASS / CAP_HELD_PER_CLASS, selecting by SORTED FILENAME so
      the selection is deterministic and a re-run reproduces it exactly
    * drops classes below EVAL_MIN_CLASS_IMAGES -- they cannot support an evaluation
    * downscales the longest edge to --max-side (288 by default; everything trains at 224, so
      storing more is ~4x the bytes for no benefit) and re-encodes at JPEG q92
    * is resumable: an image already present in the output is not redone

    python scripts/prepare_upload.py --src "C:/kaggle/working/exp_data" --out "C:/kaggle/upload"
    python scripts/prepare_upload.py --dry-run          # plan and size estimate only
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def plan(src: Path, min_images: int) -> tuple[dict, list]:
    """Which classes survive, and which files are kept for each. Deterministic."""
    want = set(C.WANT_CROPS)
    held = set(C.HELDOUT_CROPS)
    keep, skipped = {}, []
    for d in sorted(src.iterdir()):
        if not d.is_dir() or "___" not in d.name:
            continue
        crop = d.name.split("___", 1)[0]
        if crop not in want:
            skipped.append((d.name, "crop not in this study"))
            continue
        files = sorted(d.glob("*.jpg"))
        if len(files) < min_images:
            skipped.append((d.name, f"only {len(files)} images (< {min_images})"))
            continue
        cap = C.CAP_HELD_PER_CLASS if crop in held else C.CAP_TRAIN_PER_CLASS
        keep[d.name] = files[:cap]
    return keep, skipped


def convert(job):
    """Downscale and re-encode one image. Returns bytes written, or 0 if it was already there."""
    src, dst, max_side = job
    if dst.exists():
        return 0
    from PIL import Image
    try:
        img = Image.open(src).convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)   # preserves aspect ratio
        tmp = dst.with_suffix(".tmp")
        img.save(tmp, format="JPEG", quality=92)
        tmp.replace(dst)          # atomic: an interrupted run never leaves a truncated JPEG
        return dst.stat().st_size
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"C:/kaggle/working/exp_data")
    ap.add_argument("--out", default=r"C:/kaggle/upload/exp_data")
    ap.add_argument("--max-side", type=int, default=288)
    ap.add_argument("--min-images", type=int, default=C.EVAL_MIN_CLASS_IMAGES)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        sys.exit(f"[fatal] source not found: {src}")

    keep, skipped = plan(src, args.min_images)
    crops = Counter(k.split("___", 1)[0] for k in keep)
    n_keep = sum(len(v) for v in keep.values())
    held = set(C.HELDOUT_CROPS)

    print(f"[plan] {len(keep)} classes over {len(crops)} crops, {n_keep:,} images "
          f"-> {out}  (max_side={args.max_side})")
    print(f"{'crop':14s} {'classes':>8s} {'images':>9s}   role")
    print("-" * 48)
    for crop, n in sorted(crops.items(), key=lambda kv: -kv[1]):
        imgs = sum(len(v) for k, v in keep.items() if k.startswith(crop + "___"))
        print(f"{crop:14s} {n:8d} {imgs:9,d}   {'HELD' if crop in held else 'seen'}")
    missing = [c for c in C.WANT_CROPS if c not in crops]
    if missing:
        print(f"\n[WARN] no usable classes for: {missing}")
    if skipped:
        drop_crop = sum(1 for _, r in skipped if r.startswith("crop"))
        print(f"\n[plan] skipped {len(skipped)} folders ({drop_crop} outside this study, "
              f"{len(skipped) - drop_crop} below the {args.min_images}-image floor)")

    if args.dry_run:
        print(f"\n[dry-run] estimated output ~{n_keep * 26 / 1e6:.1f} GB at 288 px. "
              f"Re-run without --dry-run to build it.")
        return

    jobs = []
    for cls, files in keep.items():
        (out / cls).mkdir(parents=True, exist_ok=True)
        jobs += [(f, out / cls / f.name, args.max_side) for f in files]

    print(f"\n[build] {len(jobs):,} images on {args.workers} workers ...", flush=True)
    written = failed = skipped_n = 0
    total_bytes = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(convert, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            n = f.result()
            if n < 0:
                failed += 1
            elif n == 0:
                skipped_n += 1
            else:
                written += 1
                total_bytes += n
            if i % 5000 == 0:
                print(f"   {i:,}/{len(jobs):,}  ({total_bytes / 1e9:.2f} GB written)", flush=True)

    on_disk = sum(f.stat().st_size for f in out.rglob("*.jpg"))
    n_final = sum(1 for _ in out.rglob("*.jpg"))
    print(f"\n[done] {n_final:,} images, {on_disk / 1e9:.2f} GB in {out}")
    print(f"       {written:,} converted, {skipped_n:,} already present, {failed:,} unreadable")

    (out.parent / "dataset_summary.json").write_text(json.dumps({
        "source": str(src), "max_side": args.max_side,
        "sage_revision": C.SHARD_REVISION,
        "classes": len(keep), "crops": sorted(crops), "images": n_final,
        "bytes": on_disk, "min_images": args.min_images,
        "cap_train": C.CAP_TRAIN_PER_CLASS, "cap_held": C.CAP_HELD_PER_CLASS,
        "per_class": {k: len(v) for k, v in sorted(keep.items())},
    }, indent=2), encoding="utf-8")

    # dataset-metadata.json is REQUIRED by `kaggle datasets create` and ignored by the web
    # uploader, so writing it costs nothing and removes a failure mode if the CLI is used.
    meta = out.parent / "dataset-metadata.json"
    if not meta.exists():
        meta.write_text(json.dumps({
            "title": "pde-sage-data",
            "id": "YOUR_KAGGLE_USERNAME/pde-sage-data",
            "licenses": [{"name": "other"}],
        }, indent=2), encoding="utf-8")

    print(f"""
NEXT -- upload {out.parent} to Kaggle as a PRIVATE dataset titled  pde-sage-data

  Web (simplest, nothing to configure):
      kaggle.com -> Datasets -> New Dataset -> drag the folder -> title: pde-sage-data

  CLI (needs ~/.kaggle/kaggle.json, and edit the id in dataset-metadata.json first):
      kaggle datasets create -p "{out.parent}"

Then paste kaggle/RUN_THIS.py into a Kaggle notebook with that dataset attached.
""")


if __name__ == "__main__":
    main()
