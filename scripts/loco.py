"""
Phase C (rebuttal-proofing) — Leave-One-Crop-Out zero-shot stability.

Kills the "you cherry-picked Coffee/Orange/Peach" reviewer attack. The backbone is frozen and never
trained on ANY crop, so the descriptor head is zero-shot for every crop equally. We therefore run the
HARD cross-crop setting over a wider pool of crops (held + normally-trained), where each image must be
matched against the disease prototypes of ALL crops in the pool, and report per-crop accuracy with a
bootstrap 95% CI. Stable accuracy across a rotating crop pool = the result is not specific to one lucky
split.

Reuses the validated zero-shot path. Inference only; runs on the 4060 or CPU.

USAGE
  PDE_DATASET_DIR=/c/kaggle/working/exp_data PDE_DATA_ROOT=/c/kaggle/working \
      python scripts/loco.py --model s0 --crops Coffee Orange Peach Apple Corn Potato
  python scripts/loco.py --model s0 --strategy grounded --bootstrap 2000
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import sage_data
import zeroshot
import descriptors as D


def bootstrap_ci(correct, b=1000, seed=0):
    """correct: list[0/1]. Returns (mean, lo, hi) 95% percentile bootstrap."""
    import numpy as np
    a = np.asarray(correct, dtype=float)
    if len(a) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(b, len(a)))].mean(1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="s0", choices=list(C.DEPLOY_MODELS) + list(C.REFERENCE_MODELS))
    ap.add_argument("--crops", nargs="+",
                    default=["Coffee", "Orange", "Peach", "Apple", "Corn", "Potato"])
    ap.add_argument("--strategy", default="rich")
    ap.add_argument("--min-images", type=int, default=15)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name, pretrained = C.resolve_models([args.model])[0]

    caps = {c: C.CAP_TRAIN_PER_CLASS for c in args.crops}
    rows = sage_data.fetch(args.crops, caps, min_held_crops=None, min_class_images=args.min_images)
    rows = [r for r in rows if r["crop"] in set(args.crops)]
    assert rows, "no images for the requested crops (set PDE_DATASET_DIR)"
    classes = sorted({r["label"] for r in rows})   # pooled label space (hard cross-crop)
    print(f"[loco] model={name}  pool={args.crops}  {len(rows):,} imgs  {len(classes)} classes  "
          f"chance={1/len(classes):.1%}  strategy={args.strategy}\n")

    model, preprocess, tok, params_m = zeroshot.load_model(name, pretrained, device)
    img_emb, labels = zeroshot.embed_images(model, preprocess, rows, device)
    protos = D.build_prototypes(model, tok, classes, args.strategy, device)
    pred = (img_emb.to(device) @ protos.T).argmax(1).cpu().tolist()

    by_crop = defaultdict(list)
    all_correct = []
    for p, gt in zip(pred, labels):
        hit = int(classes[p] == gt)
        by_crop[gt.split("|")[0]].append(hit)
        all_correct.append(hit)

    out = {"model": f"{name}/{pretrained}", "img_params_M": round(params_m, 2),
           "n_classes": len(classes), "chance": 1 / len(classes),
           "strategy": args.strategy, "pool": args.crops, "per_crop": {}}
    print(f"  {'crop':10s} {'n':>5s}  acc   [95% CI]      role")
    for crop in sorted(by_crop):
        mean, lo, hi = bootstrap_ci(by_crop[crop], b=args.bootstrap)
        role = "held" if crop in C.HELDOUT_CROPS else ("train-pool" if crop in C.TRAIN_CROPS else "other")
        out["per_crop"][crop] = {"n": len(by_crop[crop]), "acc": round(mean, 4),
                                 "ci95": [round(lo, 4), round(hi, 4)], "role": role}
        print(f"  {crop:10s} {len(by_crop[crop]):5d}  {mean:.1%}  [{lo:.1%}, {hi:.1%}]  {role}")
    mean, lo, hi = bootstrap_ci(all_correct, b=args.bootstrap)
    out["pooled"] = {"n": len(all_correct), "acc": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    print(f"  {'POOLED':10s} {len(all_correct):5d}  {mean:.1%}  [{lo:.1%}, {hi:.1%}]")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = C.RESULTS_DIR / f"loco_{args.model}_{args.strategy}.json"
    # LOCO runs on a single descriptor strategy (default `rich`), so a file produced before the
    # matcher fix is invalid end to end, not just in one column. Stamp it so the runner recomputes
    # rather than skipping on "the result file exists".
    out = {"matcher_normalised": True, **out}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[loco] saved {out_path}")
    print("[loco] stable per-crop accuracy across held + train-pool crops = not a cherry-picked split.")


if __name__ == "__main__":
    main()
