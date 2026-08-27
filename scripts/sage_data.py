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
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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



def setup_hf_env(verbose: bool = True) -> bool:
    """Authenticate and accelerate the HF transport. Returns True if a token was found.

    An unauthenticated puller gets a much lower rate limit, and once throttled the connection does
    not fail cleanly -- it stalls. That is exactly what killed the 12 h Kaggle run: shard 0002 was
    requested at t+548 s and had still not produced a single byte when the notebook was terminated
    11.8 hours later. Token first, hard timeouts second (see download_shard)."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    elif verbose:
        print("[hf] WARNING: no HF_TOKEN. Anonymous pulls are rate-limited and stall under load; "
              "set a read token to make this reliable.")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")   # fail a dead socket instead of hanging
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        if verbose:
            print("[hf] hf_transfer enabled (parallel range requests)")
    except Exception:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)    # never leave it on without the package
    return bool(tok)


# Downloading in a CHILD PROCESS is the only way to enforce a real deadline: a stalled
# hf_hub_download cannot be interrupted inside its own thread, and huggingface_hub's internal
# backoff will happily retry a throttled endpoint for hours. A child can simply be killed.
_DL_SRC = """
import os, sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id=sys.argv[1], repo_type="dataset",
                      filename=sys.argv[2], revision=sys.argv[3]))
"""


def download_shard(si: int, timeout: int = 900, retries: int = 3) -> Path | None:
    """Fetch one parquet shard with a hard wall-clock deadline. None if it never arrives."""
    fn = C.SHARD_FILENAME.format(si=si)     # layout differs between the May and August releases
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-c", _DL_SRC, C.SAGE_HF_REPO, fn, C.SHARD_REVISION],
                               capture_output=True, text=True, timeout=timeout, env=os.environ.copy())
            if r.returncode == 0:
                path = Path(r.stdout.strip().splitlines()[-1])
                if path.exists():
                    print(f"    shard {si:04d} downloaded in {time.time() - t0:.0f}s", flush=True)
                    return path
            err = (r.stderr or "").strip().splitlines()[-1:] or ["no output"]
            print(f"    !! shard {si:04d} attempt {attempt}/{retries}: {err[0][:120]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"    !! shard {si:04d} attempt {attempt}/{retries}: STALLED past {timeout}s, killed",
                  flush=True)
        if attempt < retries:
            time.sleep(5 * attempt)
    print(f"    !! shard {si:04d} GIVING UP after {retries} attempts", flush=True)
    return None


def free_shard(path: Path) -> None:
    """hf_hub_download returns a SYMLINK under snapshots/ pointing at the payload in blobs/.
    Unlinking only the symlink frees nothing, so a multi-shard fetch silently accumulates the whole
    branch and fills the disk. Resolve to the blob and delete that too."""
    try:
        blob = path.resolve()
        path.unlink(missing_ok=True)
        if blob.exists() and blob != path:
            blob.unlink()
    except Exception as e:
        print(f"    [warn] could not free {path.name}: {type(e).__name__}: {e}")


def fetch(crops, caps, min_held_crops=None, min_class_images=None, max_side=None,
          budget_h=None, shard_timeout=900):
    """Download shards until `crops` are covered. `caps` = {crop: per-class cap}. Stops when
    >= min_held_crops of `crops` have >= min_class_images (None -> require ALL crops).

    max_side: if set, downscale each image so its longest edge is at most this many pixels before
    saving. SAGE images arrive at 448px and larger while every model here trains and evaluates at
    C.IMG_SIZE (224), so full-resolution storage is ~4x larger for no benefit. This matters on Kaggle,
    where /kaggle/working is ~20 GB and the full-resolution subset is 19.8 GB -- i.e. it does not fit.
    Downscaling is NOT applied to already-downloaded images, so a directory built with and without it
    would be inconsistent; use one setting per dataset build.

    budget_h: stop starting new shards after this many hours and return what is on disk. A Kaggle
    cell that overruns the 12 h limit is KILLED and commits nothing, so a fetch that politely stops
    at 8 h is worth vastly more than one that is 95%% done at 12 h. Progress is checkpointed per
    shard, so the next run resumes exactly where this one stopped.
    shard_timeout: per-shard wall-clock deadline in seconds before the download is killed+retried."""
    import pyarrow.parquet as pq
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

    setup_hf_env()
    t_start = time.time()
    done = _load_done()
    kept = Counter((r["crop"], r["disease"]) for r in rows)
    hashes = {Path(r["path"]).stem for r in rows}

    todo = [s for s in C.SHARD_ORDER[:C.MAX_SHARDS] if s not in done]
    print(f"[sage] fetching {crops} from SAGE shards "
          f"(have {len(rows):,} imgs; {len(done)}/{C.MAX_SHARDS} shards done, {len(todo)} to go)")

    # Decode of shard N overlaps the download of shard N+1: decoding is CPU-bound and downloading
    # is network-bound, and serialising them was costing roughly a third of the fetch.
    pool = ThreadPoolExecutor(max_workers=1)
    ahead = pool.submit(download_shard, todo[0], shard_timeout) if todo else None

    for i, si in enumerate(todo):
        if covered(rows):
            print("[sage] coverage target reached -> stopping early.")
            break
        if budget_h is not None and (time.time() - t_start) / 3600 > budget_h:
            print(f"[sage] BUDGET: {budget_h} h reached with {len(todo) - i} shard(s) left "
                  f"({', '.join(f'{s:04d}' for s in todo[i:][:6])}...). Stopping cleanly so the "
                  f"session commits -- re-run to resume from here.", flush=True)
            break

        path = ahead.result() if ahead is not None else download_shard(si, shard_timeout)
        ahead = (pool.submit(download_shard, todo[i + 1], shard_timeout)
                 if i + 1 < len(todo) else None)
        if path is None:
            continue                       # unreachable shard: skip, do NOT mark done, retry next run
        pf = pq.ParquetFile(str(path))
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
        free_shard(path)
        done.add(si); _save_done(done)
        by = Counter(r["crop"] for r in rows)
        el = (time.time() - t_start) / 60
        print(f"    [{i + 1}/{len(todo)} t+{el:.0f}m] after shard {si:04d}: {len(rows):,} imgs | "
              + ", ".join(f"{c}={by.get(c, 0)}" for c in crops), flush=True)

    pool.shutdown(wait=False)
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
    ap.add_argument("--budget-h", type=float, default=None,
                    help="stop starting new shards after N hours and exit cleanly. On Kaggle always "
                         "set this below the session limit: an overrun is killed and commits nothing.")
    ap.add_argument("--shard-timeout", type=int, default=900,
                    help="seconds before a stalled shard download is killed and retried")
    args = ap.parse_args()
    kw = dict(max_side=args.max_side, budget_h=args.budget_h, shard_timeout=args.shard_timeout)
    if args.role == "heldout":
        fetch(C.HELDOUT_CROPS, full_caps(), min_held_crops=C.MIN_HELD_CROPS, **kw)
    elif args.role == "train":
        fetch(C.TRAIN_CROPS, full_caps(), min_held_crops=4, **kw)
    else:
        fetch(C.WANT_CROPS, full_caps(), min_held_crops=None, **kw)


if __name__ == "__main__":
    main()
