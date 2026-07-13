"""
Evaluate — the end-to-end Phase-0/A driver (frozen model family x descriptor strategy).

Recreates the validated process in one run:
  1. fetch held-out crop images from SAGE (skipped if a Kaggle Dataset is attached);
  2. for each model (MODEL_TIERS + TEACHERS) and each descriptor strategy, measure cross-crop
     zero-shot accuracy;
  3. save results to <RESULTS_DIR>/zeroshot_eval.json  (feeds docs/paper/make_figures.py).

Run on Kaggle (GPU, Internet ON) after cloning the repo:
    PDE_DATA_ROOT=/kaggle/working python scripts/evaluate.py
    python scripts/evaluate.py --strategies bare crude rich --tiers small mid large --teachers
"""
from __future__ import annotations
import argparse
import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import sage_data
import zeroshot


def ensure_deps():
    need = []
    for mod, pkg in [("open_clip", "open_clip_torch>=2.24"), ("timm", "timm>=1.0.3"),
                     ("huggingface_hub", "huggingface_hub"), ("pyarrow", "pyarrow"), ("tqdm", "tqdm")]:
        try:
            importlib.import_module(mod)
        except Exception:
            need.append(pkg)
    if need:
        print(f"[deps] installing {need}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=["bare", "crude", "rich"])
    ap.add_argument("--tiers", nargs="+", default=list(C.MODEL_TIERS))
    ap.add_argument("--heavy", action="store_true", help="also evaluate the heavyweight (~86M) tier")
    ap.add_argument("--teachers", action="store_true", help="also evaluate the reference/teacher VLMs")
    ap.add_argument("--exp", choices=list(C.EXPERIMENTS), default="C",
                    help="scale-study subset: A (3 held) / B (6) / C (8, all)")
    args = ap.parse_args()

    ensure_deps()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    held_crops = C.EXPERIMENTS[args.exp]["held"]
    rows = sage_data.fetch(C.HELDOUT_CROPS, sage_data.full_caps(), min_held_crops=C.MIN_HELD_CROPS)
    rows = [r for r in rows if r["crop"] in set(held_crops)]     # subset to this experiment's held crops
    assert rows, f"no held-out images for experiment {args.exp} ({held_crops})"
    classes = sorted({r["label"] for r in rows})
    print(f"[eval] experiment {args.exp}: held crops = {held_crops}")
    chance = 1.0 / len(classes)
    crops = sorted({c.split("|")[0] for c in classes})
    print(f"[eval] held={len(rows):,} imgs  {len(classes)} classes  crops={crops}  chance={chance:.1%}\n")

    models = [(C.MODEL_TIERS[t][0], C.MODEL_TIERS[t][1]) for t in args.tiers if t in C.MODEL_TIERS]
    if args.heavy:
        models.append(C.HEAVYWEIGHT)
    if args.teachers:
        models += [m for m in C.TEACHERS if m not in models]

    results, coverage = {}, {}
    for name, pretrained in models:
        try:
            cache = None
            row = {}
            for strat in args.strategies:
                res, cache = zeroshot.evaluate(name, pretrained, rows, classes, strat, device,
                                               reuse_img_emb=cache)
                row[strat] = res
            results[f"{name}/{pretrained}"] = row
            cols = "  ".join(f"{s}={row[s]['acc']:.1%}" for s in args.strategies)
            print(f"  {name:20s} img={row[args.strategies[0]]['img_params_M']:6.1f}M  {cols}")
        except Exception as e:
            print(f"  {name:20s} skipped ({type(e).__name__}: {str(e)[:60]})")

    # record rich-descriptor coverage (which classes matched a rich descriptor)
    import descriptors as D
    for c in classes:
        D.text_for(c, "rich", coverage)

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS_DIR / f"zeroshot_eval_{args.exp}.json"
    out.write_text(json.dumps({"chance": chance, "n_classes": len(classes), "crops": crops,
                               "coverage": coverage, "models": results}, indent=2))
    print(f"\n[eval] saved {out}")


if __name__ == "__main__":
    main()
