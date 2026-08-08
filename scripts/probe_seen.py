"""
Seen-crop linear probe for the frozen VLM tiers — the VLM half of the baseline table.

Complements supervised_baseline.py (full-CNN training). Here we train a linear probe on FROZEN VLM
image features for each deploy tier (s0/s1/s2/b) on the seen crops. This is the "same backbone that
does unseen zero-shot, now measured on seen crops" number: it shows the VLM route is competitive with
supervised CNNs on seen WHILE also generalizing to unseen (which the CNNs structurally cannot).

Usage:
  python scripts/probe_seen.py --models s0 s1 s2 b --exp C
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import sage_data
import zeroshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(C.DEPLOY_MODELS))
    ap.add_argument("--exp", choices=list(C.EXPERIMENTS), default="C")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seen_crops = C.EXPERIMENTS[args.exp]["seen"]
    rows = sage_data.fetch(C.TRAIN_CROPS, sage_data.full_caps(), min_held_crops=C.MIN_HELD_CROPS)
    rows = [r for r in rows if r["crop"] in set(seen_crops)]
    assert rows, f"no seen images for experiment {args.exp} ({seen_crops})"
    classes = sorted({r["label"] for r in rows})
    cidx = {c: i for i, c in enumerate(classes)}
    print(f"[probe] exp {args.exp}: {len(rows):,} seen imgs, {len(classes)} classes, crops={seen_crops}")

    # seen_images/n_seen_crops are recorded, not just printed: the class count depends on which SAGE
    # shards happen to be on disk, so two runs of the same --exp on different days are NOT comparable
    # without it. (probe_seen_C.json from 14 Jul 2026 had 166 classes; the pool has grown since.)
    out = {"exp": args.exp, "seen_classes": len(classes), "seen_images": len(rows),
           "n_seen_crops": len(seen_crops), "seen_crops": seen_crops, "models": {}}
    for name, pretrained in C.resolve_models(args.models):
        model, preprocess, tok, params_m = zeroshot.load_model(name, pretrained, device)
        emb, labels = zeroshot.embed_images(model, preprocess, rows, device)
        random.seed(C.RANDOM_SEED)
        by = defaultdict(list)
        for i, l in enumerate(labels):
            by[l].append(i)
        tr, te = [], []
        for l, idxs in by.items():
            random.shuffle(idxs); k = max(1, int(0.2 * len(idxs))); te += idxs[:k]; tr += idxs[k:]
        Xtr = emb[tr].to(device); ytr = torch.tensor([cidx[labels[i]] for i in tr], device=device)
        Xte = emb[te].to(device); yte = [labels[i] for i in te]
        clf = nn.Linear(emb.shape[1], len(classes)).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(args.epochs):
            perm = torch.randperm(len(tr), device=device)
            for s in range(0, len(perm), 256):
                b = perm[s:s + 256]
                lo = F.cross_entropy(clf(Xtr[b]), ytr[b]); opt.zero_grad(); lo.backward(); opt.step()
        with torch.no_grad():
            pred = clf(Xte).argmax(1).cpu().tolist()
        acc = sum(classes[p] == gt for p, gt in zip(pred, yte)) / len(yte)
        out["models"][name] = {"img_params_M": round(params_m, 2), "seen_probe_top1": round(acc, 4)}
        print(f"  {name:16s} {params_m:6.1f}M  seen linear-probe top1 = {acc:.1%}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = C.RESULTS_DIR / f"probe_seen_{args.exp}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[probe] the VLM tiers do seen (probe) AND unseen (zero-shot); the CNNs do seen only.")
    print(f"[probe] saved {p}")


if __name__ == "__main__":
    main()
