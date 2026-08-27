"""
Phase C3 metrics — top-1, top-5, and the ABSTAIN / risk-coverage curve.

Why: fine-grained 17-class cross-crop zero-shot top-1 is modest (~27%). A field tool doesn't have
to answer every image — it can ABSTAIN when unsure. Reporting top-5 and a risk-coverage curve
(accuracy vs. fraction answered) is what makes the headline number defensible for a reviewer.

Confidence = max cosine similarity to any descriptor prototype (protos & image embs are L2-normed).
We sort predictions by confidence and, for each coverage level, report the accuracy on the answered
fraction. AURC (area under the risk-coverage curve, lower is better) summarizes it in one number.

Reuses the validated zero-shot path (scripts/zeroshot.py + descriptors.py). Held data comes from the
same SAGE fetch as everything else. Runs on the 4060 (or CPU) — inference only.

USAGE
  PDE_DATASET_DIR=/c/kaggle/working/exp_data PDE_DATA_ROOT=/c/kaggle/working \
      python scripts/metrics.py --models s0 s1 s2 b --strategies rich grounded
  python scripts/metrics.py --models s0 --strategies grounded --reference   # + SigLIP2 ceiling
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import sage_data
import zeroshot
import descriptors as D


def topk_and_riskcoverage(sims, labels, classes, ks=(1, 5),
                          coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """sims: [N, n_classes] cosine similarities. Returns dict with top-k accs, AURC,
    selective accuracy at each coverage, and the full risk-coverage curve."""
    import torch
    n, ncls = sims.shape
    gt = torch.tensor([classes.index(l) for l in labels])
    ks = tuple(k for k in ks if k <= ncls)
    topk = {}
    for k in ks:
        pred_k = sims.topk(k, dim=1).indices            # [N, k]
        topk[f"top{k}"] = (pred_k == gt[:, None]).any(1).float().mean().item()

    # Two confidence signals for abstention: raw max-similarity vs the top1-top2 MARGIN.
    # Margin is usually far better calibrated for CLIP zero-shot (max-sim is near-constant / anti-
    # calibrated). We report the risk-coverage under margin (primary) and max-sim AURC for comparison.
    top2 = sims.topk(2, dim=1).values
    pred1 = sims.argmax(1)
    correct = (pred1 == gt).float()
    maxsim = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1]

    def _rc(conf):
        order = torch.argsort(conf, descending=True)     # most-confident kept first
        cum_acc = torch.cumsum(correct[order], 0) / torch.arange(1, n + 1)
        aurc = (1.0 - cum_acc).mean().item()             # area under risk-coverage (lower = better)
        sel = {f"acc@cov{int(c*100)}": cum_acc[max(1, int(round(c * n))) - 1].item() for c in coverages}
        curve = [[round((i + 1) / n, 4), round(cum_acc[i].item(), 4)]
                 for i in range(0, n, max(1, n // 100))]
        return aurc, sel, curve

    aurc_margin, sel_margin, curve_margin = _rc(margin)
    aurc_maxsim, _, _ = _rc(maxsim)
    return {**{k: round(v, 4) for k, v in topk.items()},
            "aurc": round(aurc_margin, 4), "aurc_maxsim": round(aurc_maxsim, 4),
            "confidence": "margin(top1-top2)",
            **{k: round(v, 4) for k, v in sel_margin.items()},
            "risk_coverage_curve": curve_margin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(C.DEPLOY_MODELS))
    ap.add_argument("--strategies", nargs="+", default=["rich", "grounded"])
    ap.add_argument("--reference", action="store_true", help="also evaluate SigLIP2 ceiling")
    ap.add_argument("--exp", choices=list(C.EXPERIMENTS), default="C",
                    help="scale-study subset: A (3 held) / B (6) / C (8, all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    held_crops = C.EXPERIMENTS[args.exp]["held"]
    rows = sage_data.fetch(C.HELDOUT_CROPS, sage_data.full_caps(), min_held_crops=C.MIN_HELD_CROPS)
    rows = [r for r in rows if r["crop"] in set(held_crops)]
    assert rows, f"no held-out images for experiment {args.exp} ({held_crops})"
    classes = sorted({r["label"] for r in rows})
    print(f"[metrics] held={len(rows):,} imgs  {len(classes)} classes  chance={1/len(classes):.1%}  device={device}\n")

    models = C.resolve_models(args.models, include_reference=args.reference)
    out = {"n_classes": len(classes), "chance": 1 / len(classes), "n_images": len(rows), "models": {}}
    for name, pretrained in models:
        try:
            model, preprocess, tok, params_m = zeroshot.load_model(name, pretrained, device)
            img_emb, labels = zeroshot.embed_images(model, preprocess, rows, device)
            row = {"img_params_M": round(params_m, 2)}
            for strat in args.strategies:
                protos = D.build_prototypes(model, tok, classes, strat, device)
                sims = (img_emb.to(device) @ protos.T).cpu()
                row[strat] = topk_and_riskcoverage(sims, labels, classes)
            out["models"][f"{name}/{pretrained}"] = row
            summ = "  ".join(
                f"{s}:top1={row[s]['top1']:.1%}/top5={row[s]['top5']:.1%}/aurc={row[s]['aurc']:.3f}"
                for s in args.strategies)
            print(f"  {name:18s} {params_m:6.1f}M  {summ}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name:18s} skipped ({type(e).__name__}: {str(e)[:70]})")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else C.RESULTS_DIR / f"metrics_abstain_{args.exp}.json"
    # Stamped because this file reports a `rich` arm, and `rich` came from a matcher that could not
    # reach any multi-word bank key. An unstamped file predates the fix and must be recomputed, not
    # skipped by a "result already exists" guard.
    if isinstance(out, dict):
        out = {"matcher_normalised": True, **out}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[metrics] saved {out_path}")
    print("[metrics] top-5 and acc@cov are the field-relevant, reviewer-defensible numbers.")


if __name__ == "__main__":
    main()
