"""
SAGE data fetch — download + filter SAGE parquet shards into a local class-folder dataset.

The WORKING data path (row-by-row HF streaming is ~2.5s/row -> unusable). We download the
auto-converted parquet shards, filter to our crops up to a per-class cap, dedupe by content
hash, and save JPEGs as `<DATASET_DIR>/<Crop>___<Disease>/<hash>.jpg`. Incremental + resumable
via a `.shards_done.json` marker, so re-runs add only missing shards.

Tip (Kaggle): build ONCE, save DATASET_DIR as a Kaggle Dataset, attach it next time -> the
build is skipped entirely and you get all crops (including the rare ones) with no re-download.

Usage (import or CLI):
    python scripts/sage_data.py --role heldout      # fetch held-out crops only (fast)
    python scripts/sage_data.py --role all          # fetch train + held-out crops
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def load_rows(crops, min_imgs: int = 1):
    """Read on-disk class folders for `crops`, return [{path, crop, disease, label}], dropping
    classes with fewer than `min_imgs` images."""
    rows = []
    if not C.DATASET_DIR.exists():
        return rows
    want = set(crops)
    for d in C.DATASET_DIR.iterdir():
        if not d.is_dir() or "___" not in d.name:
            continue
        crop, disease = d.name.split("___", 1)
        if crop not in want:
            continue
        for jpg in d.glob("*.jpg"):
            rows.append({"path": str(jpg), "crop": crop, "disease": disease, "label": f"{crop}|{disease}"})
    cc = Counter(r["label"] for r in rows)
    return [r for r in rows if cc[r["label"]] >= min_imgs]


def full_caps():
    """Caps for ALL wanted crops (train + held). A shard download mines every wanted crop, so a
    held-only fetch still saves train crops (and vice-versa) -> one shared, valid .shards_done.json."""
    return {**{c: C.CAP_TRAIN_PER_CLASS for c in C.TRAIN_CROPS},
            **{c: C.CAP_HELD_PER_CLASS for c in C.HELDOUT_CROPS}}


def _done_path() -> Path:
    return C.DATASET_DIR / ".shards_done.json"


def _load_done():
    p = _done_path()
    if p.exists():
        try:
            return set(json.loads(p.read_text()))
        except Exception:
            return set()
    return set()


def _save_done(done):
    _done_path().write_text(json.dumps(sorted(done)))


def fetch(crops, caps, min_held_crops=None, min_class_images=None, max_side=None):
    """Download shards until `crops` are covered. `caps` = {crop: per-class cap}. Stops when
    >= min_held_crops of `crops` have >= min_class_images (None -> require ALL crops).

    max_side: if set, downscale each image so its longest edge is at most this many pixels before
    saving. SAGE images arrive at 448px and larger while every model here trains and evaluates at
    C.IMG_SIZE (224), so full-resolution storage is ~4x larger for no benefit. This matters on Kaggle,
    where /kaggle/working is ~20 GB and the full-resolution subset is 19.8 GB -- i.e. it does not fit.
    Downscaling is NOT applied to already-downloaded images, so a directory built with and without it
    would be inconsistent; use one setting per dataset build."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from tqdm.auto import tqdm

    min_class_images = min_class_images or C.EVAL_MIN_CLASS_IMAGES

    def covered(rows):
        by = Counter(r["crop"] for r in rows)
        n = sum(1 for c in crops if by.get(c, 0) >= min_class_images)
        need = len(crops) if min_held_crops is None else min_held_crops
        return n >= need

    C.ensure_dirs()
    rows = load_rows(list(caps.keys()), min_imgs=1)   # seed from ALL wanted crops on disk
    if covered(rows):
        print(f"[sage] {len(rows):,} imgs already on disk for {crops} -> skip download.")
        return load_rows(crops, min_imgs=min_class_images)

    done = _load_done()
    kept = Counter((r["crop"], r["disease"]) for r in rows)
    hashes = {Path(r["path"]).stem for r in rows}
    print(f"[sage] fetching {crops} from SAGE shards (have {len(rows):,}) ...")
    for si in C.SHARD_ORDER[:C.MAX_SHARDS]:
        if covered(rows):
            break
        if si in done:
            continue
        fn = f"default/train/{si:04d}.parquet"
        print(f"    downloading shard {si:04d} ...")
        try:
            path = hf_hub_download(repo_id=C.SAGE_HF_REPO, repo_type="dataset",
                                   filename=fn, revision=C.SHARD_REVISION)
        except Exception as e:
            print(f"    !! shard {si:04d} failed: {e}")
            continue
        pf = pq.ParquetFile(path)
        try:
            names = set(pf.schema_arrow.names)
            cols = [c for c in ("image", "crop", "disease") if c in names]
        except Exception:
            cols = None
        for batch in tqdm(pf.iter_batches(batch_size=512, columns=cols),
                          total=pf.metadata.num_rows // 512 + 1, desc=f"shard{si:04d}"):
            d = batch.to_pydict()
            imgs = d.get("image", []); crps = d.get("crop", [])
            diss = d.get("disease", [None] * len(crps))
            for img_obj, craw, draw in zip(imgs, crps, diss):
                crop = C.canonical_crop(craw)
                if crop is None or crop not in caps:     # keep ALL wanted crops, not just `crops`
                    continue
                cap = caps.get(crop, C.CAP_HELD_PER_CLASS)
                disease = str(draw if draw is not None else "Unknown")
                key = (crop, disease)
                if kept[key] >= cap:
                    continue
                try:
                    raw = img_obj["bytes"] if isinstance(img_obj, dict) else img_obj
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    if max_side and max(img.size) > max_side:
                        img.thumbnail((max_side, max_side), Image.LANCZOS)  # preserves aspect ratio
                    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=92)
                    jpg = buf.getvalue()
                except Exception:
                    continue
                h16 = hashlib.sha1(jpg).hexdigest()[:16]
                if h16 in hashes:
                    continue
                hashes.add(h16)
                cls = C.DATASET_DIR / f"{C.safe_name(crop)}___{C.safe_name(disease)}"
                cls.mkdir(parents=True, exist_ok=True)
                (cls / f"{h16}.jpg").write_bytes(jpg)
                kept[key] += 1
                rows.append({"path": str(cls / f"{h16}.jpg"), "crop": crop,
                             "disease": disease, "label": f"{crop}|{disease}"})
        # Free the shard properly. hf_hub_download returns a SYMLINK in the snapshots/ tree pointing
        # at the real payload under blobs/; unlinking only the symlink frees nothing, so a multi-shard
        # fetch silently accumulated ~10 GB per shard and filled the disk (this is what made Kaggle
        # runs die partway through a fetch). Resolve to the blob and delete that too.
        try:
            p = Path(path)
            blob = p.resolve()
            p.unlink(missing_ok=True)
            if blob.exists() and blob != p:
                blob.unlink()
        except Exception as e:
            print(f"    [warn] could not free shard {si:04d}: {type(e).__name__}: {e}")
        done.add(si); _save_done(done)
        by = Counter(r["crop"] for r in rows)
        print(f"    after shard {si:04d}: " + ", ".join(f"{c}={by.get(c,0)}" for c in crops))

    by = Counter(r["crop"] for r in rows)
    print("[sage] counts: " + ", ".join(f"{c}={by.get(c,0)}" for c in crops))
    return load_rows(crops, min_imgs=min_class_images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["heldout", "train", "all"], default="heldout")
    ap.add_argument("--max-side", type=int, default=None,
                    help="downscale longest edge to N px before saving (e.g. 256). Everything trains "
                         "and evaluates at 224, so full-res storage is ~4x larger for no benefit; "
                         "required to fit inside Kaggle's ~20 GB /kaggle/working.")
    args = ap.parse_args()
    if args.role == "heldout":
        fetch(C.HELDOUT_CROPS, full_caps(), min_held_crops=C.MIN_HELD_CROPS, max_side=args.max_side)
    elif args.role == "train":
        fetch(C.TRAIN_CROPS, full_caps(), min_held_crops=4, max_side=args.max_side)
    else:
        fetch(C.WANT_CROPS, full_caps(), min_held_crops=None, max_side=args.max_side)


if __name__ == "__main__":
    main()
