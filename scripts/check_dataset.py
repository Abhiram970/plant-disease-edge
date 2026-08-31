"""
Governance check for the image set: is everything PRESENT, and is it INTACT?

Runs anywhere -- on this machine against C:/kaggle/upload/exp_data, or inside a Kaggle notebook
against the attached dataset, which it locates at any depth (Kaggle does not mount at a fixed
depth: /kaggle/input/datasets/<owner>/<dataset>/ is as likely as /kaggle/input/<dataset>/).

    python scripts/check_dataset.py                          # auto-locate
    python scripts/check_dataset.py --dir C:/kaggle/upload/exp_data
    python scripts/check_dataset.py --deep                   # decode EVERY image (slow, thorough)
    python scripts/check_dataset.py --sample 3000            # decode N random images (default 500)
    python scripts/check_dataset.py --manifest out.json      # write a manifest for later comparison
    python scripts/check_dataset.py --against out.json       # diff against a manifest

Exit code 0 = pass, 1 = at least one FAIL. Every check prints PASS/FAIL/WARN with the actual
number, because a governance check that says only "ok" is not evidence of anything.

PRESENCE is checked against scripts/config.py, not against hardcoded numbers, so it keeps
telling the truth if the crop lists ever change.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FAILED = []
WARNED = []


def check(ok, label, detail="", warn_only=False):
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    if not ok:
        (WARNED if warn_only else FAILED).append(f"{label}: {detail}")
    print(f"  [{tag}] {label}" + (f"  -- {detail}" if detail else ""))
    return ok


def locate(explicit=None):
    """The image root: --dir, else PDE_DATASET_DIR, else the first plausible place we can find."""
    import os
    if explicit:
        return Path(explicit)
    env = os.environ.get("PDE_DATASET_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    roots = [Path("/kaggle/input"), Path("C:/kaggle/upload"), Path("C:/kaggle/working"),
             C.DATASET_DIR, C.REPO_ROOT / "data"]
    for root in roots:
        if not root.exists():
            continue
        frontier = [(root, 0)]
        while frontier:                       # breadth-first: shallowest match wins
            d, depth = frontier.pop(0)
            if depth > 5:
                continue
            try:
                if any(d.glob("*___*")):
                    return d
                frontier.extend((p, depth + 1) for p in sorted(d.iterdir()) if p.is_dir())
            except Exception:
                continue
    return None


def scan(root):
    """{class_name: [jpg paths]} plus anything that does not belong."""
    classes, strays = {}, []
    for d in sorted(root.iterdir()):
        if d.is_file():
            strays.append(str(d))
            continue
        if "___" not in d.name:
            strays.append(str(d) + "/  (directory, not a <Crop>___<Disease> class)")
            continue
        jpgs, others = [], []
        for f in d.rglob("*"):
            if f.is_dir():
                continue
            (jpgs if f.suffix.lower() == ".jpg" else others).append(f)
        classes[d.name] = jpgs
        strays.extend(str(o) for o in others)
    return classes, strays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--sample", type=int, default=500, help="images to fully decode (0 to skip)")
    ap.add_argument("--deep", action="store_true", help="decode EVERY image instead of a sample")
    ap.add_argument("--manifest", default=None, help="write a manifest JSON here")
    ap.add_argument("--against", default=None, help="compare against a manifest written earlier")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    root = locate(args.dir)
    if root is None or not root.is_dir():
        sys.exit("[fatal] no image directory found. Pass --dir, or set PDE_DATASET_DIR.")
    print(f"\nDATASET GOVERNANCE CHECK\n{'=' * 72}\nroot: {root}\n")

    classes, strays = scan(root)
    per_class = {k: len(v) for k, v in classes.items()}
    n_images = sum(per_class.values())
    crops = Counter(k.split("___", 1)[0] for k in classes)

    # ---------------------------------------------------------------- PRESENCE
    print("PRESENCE")
    held, seen = set(C.HELDOUT_CROPS), set(C.TRAIN_CROPS)
    check(len(classes) > 0, "class folders found", f"{len(classes)} classes, {n_images:,} images")

    missing_crops = [c for c in C.WANT_CROPS if c not in crops]
    check(not missing_crops, "every configured crop present",
          f"missing: {missing_crops}" if missing_crops else f"all {len(C.WANT_CROPS)} present")

    extra = [c for c in crops if c not in set(C.WANT_CROPS)]
    check(not extra, "no crops outside the study set", f"unexpected: {extra}" if extra else "none")

    for exp in sorted(C.EXPERIMENTS):
        h = set(C.EXPERIMENTS[exp]["held"])
        n = sum(1 for k in classes if k.split("___", 1)[0] in h)
        print(f"         experiment {exp}: {len(h)} held crops -> {n} classes")
    n_seen = sum(1 for k in classes if k.split("___", 1)[0] in seen)
    n_held = sum(1 for k in classes if k.split("___", 1)[0] in held)
    check(n_seen + n_held == len(classes), "seen/held partition covers every class",
          f"{n_seen} seen + {n_held} held = {n_seen + n_held} of {len(classes)}")

    floor = C.EVAL_MIN_CLASS_IMAGES
    under = {k: v for k, v in per_class.items() if v < floor}
    check(not under, f"every class >= {floor} images (EVAL_MIN_CLASS_IMAGES)",
          f"{len(under)} under floor: {sorted(under.items())[:5]}" if under
          else f"min {min(per_class.values())}, max {max(per_class.values())}")

    empty = [k for k, v in per_class.items() if v == 0]
    check(not empty, "no empty class folders", f"{len(empty)}: {empty[:5]}" if empty else "none")

    # ---------------------------------------------------------------- INTEGRITY
    print("\nINTEGRITY")
    check(not strays, "no stray non-jpg files",
          f"{len(strays)} found: {strays[:3]}" if strays else "none")

    all_jpgs = [p for v in classes.values() for p in v]
    zero = [p for p in all_jpgs if p.stat().st_size == 0]
    check(not zero, "no zero-byte files", f"{len(zero)}: {[str(z) for z in zero[:3]]}"
          if zero else "none")

    tiny = [p for p in all_jpgs if 0 < p.stat().st_size < 1024]
    check(not tiny, "no suspiciously small files (<1 KB)",
          f"{len(tiny)} under 1 KB" if tiny else "none", warn_only=True)

    names = Counter(p.name for p in all_jpgs)
    dupes = [n for n, c in names.items() if c > 1]
    check(not dupes, "no duplicate filenames across classes",
          f"{len(dupes)} repeated (content-hash names should be unique)" if dupes else "none",
          warn_only=True)

    # decode: the only check that proves a file is actually readable
    targets = all_jpgs if args.deep else random.Random(args.seed).sample(
        all_jpgs, min(args.sample, len(all_jpgs))) if args.sample else []
    if targets:
        try:
            from PIL import Image, ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = False      # so truncation RAISES instead of passing
            bad, edges = [], Counter()
            for p in targets:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:            # verify() invalidates; reopen to load
                        im.load()
                        edges[max(im.size)] += 1
                except Exception as e:
                    bad.append((str(p), type(e).__name__))
            check(not bad, f"all {len(targets):,} decoded images are readable"
                  + (" (FULL set)" if args.deep else " (sample)"),
                  f"{len(bad)} unreadable: {bad[:3]}" if bad else "0 failures")
            over = sum(n for e, n in edges.items() if e > C.IMG_SIZE * 2)
            check(over == 0, f"longest edge <= {C.IMG_SIZE * 2} px",
                  f"{over} images exceed it" if over else f"max {max(edges)} px")
        except ImportError:
            print("  [WARN] Pillow not installed -- decode check skipped")

    total_bytes = sum(p.stat().st_size for p in all_jpgs)
    print(f"\n  size on disk: {total_bytes / 1e9:.2f} GB across {n_images:,} images")

    # ---------------------------------------------------------------- MANIFEST
    man = {"root": str(root), "images": n_images, "classes": len(classes),
           "crops": sorted(crops), "per_class": dict(sorted(per_class.items())),
           "bytes": total_bytes, "sage_revision": C.SHARD_REVISION}
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(man, indent=2), encoding="utf-8")
        print(f"  manifest written: {args.manifest}")

    if args.against:
        print("\nCOMPARISON")
        old = json.loads(Path(args.against).read_text(encoding="utf-8"))
        op, np_ = old.get("per_class", {}), man["per_class"]
        gone = sorted(set(op) - set(np_))
        added = sorted(set(np_) - set(op))
        shrunk = {k: (op[k], np_[k]) for k in set(op) & set(np_) if np_[k] < op[k]}
        check(not gone, "no class disappeared", f"{len(gone)}: {gone[:5]}" if gone else "none")
        check(not shrunk, "no class lost images",
              f"{len(shrunk)}: {list(shrunk.items())[:3]}" if shrunk else "none")
        check(not added, "no unexpected new class",
              f"{len(added)}: {added[:5]}" if added else "none", warn_only=True)
        check(old.get("sage_revision") == man["sage_revision"], "same SAGE revision",
              f"{old.get('sage_revision','?')[:12]} -> {man['sage_revision'][:12]}")

    # ---------------------------------------------------------------- VERDICT
    print(f"\n{'=' * 72}")
    if FAILED:
        print(f"FAILED ({len(FAILED)}):")
        for f in FAILED:
            print(f"  - {f}")
    if WARNED:
        print(f"WARNINGS ({len(WARNED)}) -- not blocking:")
        for w in WARNED:
            print(f"  - {w}")
    if not FAILED:
        print(f"ALL CHECKS PASSED -- {n_images:,} images, {len(classes)} classes, "
              f"{len(crops)} crops, {total_bytes / 1e9:.2f} GB")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
