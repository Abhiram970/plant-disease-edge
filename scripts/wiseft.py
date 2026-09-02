"""
WiSE-FT: the seen/unseen trade-off as a single interpolation weight.

Fine-tune the image encoder on the SEEN crops, then evaluate an interpolation between the
frozen and fine-tuned visual weights:

    theta(alpha) = (1 - alpha) * theta_frozen + alpha * theta_finetuned

at alpha = 0, 0.25, 0.5, 0.75, 1. alpha=0 must reproduce the frozen baseline exactly, which is
the built-in correctness check; alpha=1 is naive fine-tuning and shows catastrophic forgetting.

WHY THIS SCRIPT EXISTS
The manuscript's WiSE-FT table came from a `run_all` pipeline that no longer exists in the repo,
so those numbers could not be regenerated and were stranded on an older image build while every
neighbouring table was re-measured. Worse, the stranded file records `unseen_classes: 17` against
`seen_classes: 166` -- the seen side came from the nested configuration C while the unseen side
came from the 17-class pilot, so a single table mixed two protocols. This script measures BOTH
sides under one protocol (configuration C: 166 seen classes, 51 unseen), and records the
protocol in the output so the mixture cannot recur.

USAGE
  python scripts/wiseft.py --model s0 --exp C --epochs 3
"""
from __future__ import annotations
import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import sage_data
import zeroshot
import descriptors


try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:
    _TorchDataset = object


class _WiseDS(_TorchDataset):
    """Module-level so DataLoader workers can pickle it.

    Defined inside main() this raised
    "AttributeError: Can't pickle local object 'main.<locals>._DS'" the moment num_workers>0,
    because worker processes are spawned and must re-import the class by qualified name.
    supervised_baseline.py already hit this and solved it the same way.
    """

    def __init__(self, items, tf, cidx):
        self.items, self.tf, self.cidx = items, tf, cidx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        r = self.items[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), self.cidx[r["label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="s0", choices=list(C.DEPLOY_MODELS))
    ap.add_argument("--exp", default="C", choices=list(C.EXPERIMENTS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers. Defaults to 0: unlike the CNN baseline, this "
                         "script has a CUDA context and a loaded model live BEFORE the loader "
                         "is built, and spawning workers around that crashed them outright "
                         "('DataLoader worker exited unexpectedly'). The dataset here is the "
                         "seen split for a few epochs, so the loader is not the bottleneck.")
    ap.add_argument("--strategy", default="grounded")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--min-images", type=int, default=15)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name, pretrained = C.DEPLOY_MODELS[args.model]
    print(f"[wiseft] {name}/{pretrained}  exp={args.exp}  device={device}")

    # ---- data: seen crops for fine-tuning, held-out crops for the zero-shot side ----------
    held_crops = set(C.EXPERIMENTS[args.exp]["held"])
    all_rows = sage_data.fetch(C.HELDOUT_CROPS, sage_data.full_caps(),
                               min_held_crops=C.MIN_HELD_CROPS)
    unseen_rows = [r for r in all_rows if r["crop"] in held_crops]
    unseen_classes = sorted({r["label"] for r in unseen_rows})

    # Same source the linear probe uses, so the seen side is directly comparable to tab_seen.
    import csv
    if not C.MANIFEST_CSV.exists():
        sys.exit(f"[wiseft] manifest not found: {C.MANIFEST_CSV} -- run build_manifest.py first")
    with open(C.MANIFEST_CSV, newline="", encoding="utf-8") as f:
        seen_rows = [{"path": r["path"], "label": f"{r['crop']}|{r['disease']}"}
                     for r in csv.DictReader(f) if r["split_role"] == "train_crop"]

    by = defaultdict(list)
    for r in seen_rows:
        by[r["label"]].append(r)
    seen_classes = sorted(l for l, rs in by.items() if len(rs) >= args.min_images)
    cidx = {c: i for i, c in enumerate(seen_classes)}
    seen_rows = [r for r in seen_rows if r["label"] in cidx]
    print(f"[wiseft] seen: {len(seen_rows):,} imgs / {len(seen_classes)} classes")
    print(f"[wiseft] unseen: {len(unseen_rows):,} imgs / {len(unseen_classes)} classes "
          f"(chance {100.0 / max(len(unseen_classes), 1):.2f}%)")

    random.seed(C.RANDOM_SEED)
    tr, te = [], []
    for l in seen_classes:
        idx = by[l][:]
        random.shuffle(idx)
        k = max(1, int(0.2 * len(idx)))
        te += idx[:k]; tr += idx[k:]

    model, preprocess, tok, params_m = zeroshot.load_model(name, pretrained, device)
    frozen_visual = copy.deepcopy(model.visual.state_dict())

    _kw = dict(num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        _kw.update(persistent_workers=True, prefetch_factor=4)
    dl_tr = DataLoader(_WiseDS(tr, preprocess, cidx), batch_size=args.batch, shuffle=True,
                       drop_last=True, **_kw)
    dl_te = DataLoader(_WiseDS(te, preprocess, cidx), batch_size=args.batch, **_kw)

    # ---- fine-tune the VISUAL tower on seen crops (linear head on top, both trained) ------
    with torch.no_grad():
        dim = model.visual(next(iter(dl_te))[0][:1].to(device)).shape[-1]
    head = nn.Linear(dim, len(seen_classes)).to(device)
    opt = torch.optim.AdamW(list(model.visual.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    _bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    _adt = torch.bfloat16 if _bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not _bf16))

    ft_loss = []
    model.train()
    for ep in range(args.epochs):
        run = nb = 0
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=_adt, enabled=(device == "cuda")):
                loss = F.cross_entropy(head(model.visual(x)), y)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            run += loss.item(); nb += 1
        ft_loss.append(round(run / max(nb, 1), 4))
        print(f"  ft epoch {ep + 1}/{args.epochs}  loss={ft_loss[-1]:.3f}", flush=True)
    model.eval()
    finetuned_visual = copy.deepcopy(model.visual.state_dict())

    # ---- sweep alpha ----------------------------------------------------------------------
    def seen_top1():
        """Seen accuracy of the CURRENT (interpolated) encoder with the frozen-fitted head."""
        correct = tot = 0
        with torch.no_grad():
            for x, y in dl_te:
                x = x.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=_adt, enabled=(device == "cuda")):
                    pred = head(model.visual(x).float()).argmax(1).cpu()
                correct += (pred == y).sum().item(); tot += len(y)
        return correct / max(tot, 1)

    # The seen-side head must be trained against the FROZEN encoder before the sweep, or the
    # alpha=0 row cannot reproduce the frozen probe -- which is the sweep's built-in correctness
    # check. In a 1-epoch smoke test the jointly-trained head gave seen=3.3% at alpha=0 against
    # a frozen probe of ~82%: the head, not the encoder, was untrained. Re-fit a fresh linear
    # head on frozen features (cheap: features are computed once) and use it for every alpha, so
    # the only thing varying across the sweep is the encoder interpolation.
    print("[wiseft] re-fitting the seen head on FROZEN features for the alpha=0 check ...",
          flush=True)
    model.visual.load_state_dict(frozen_visual)
    model.eval()
    with torch.no_grad():
        Xtr, Ytr, Xte, Yte = [], [], [], []
        for dl, X, Y in ((dl_tr, Xtr, Ytr), (dl_te, Xte, Yte)):
            for x, y in dl:
                x = x.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=_adt, enabled=(device == "cuda")):
                    X.append(model.visual(x).float().cpu())
                Y.append(y)
        Xtr = torch.cat(Xtr); Ytr = torch.cat(Ytr)
        Xte = torch.cat(Xte); Yte = torch.cat(Yte)
    head = nn.Linear(Xtr.shape[1], len(seen_classes)).to(device)
    hopt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ep in range(20):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            xb = Xtr[b].to(device); yb = Ytr[b].to(device)
            hopt.zero_grad(set_to_none=True)
            F.cross_entropy(head(xb), yb).backward(); hopt.step()
    with torch.no_grad():
        frozen_probe = (head(Xte.to(device)).argmax(1).cpu() == Yte).float().mean().item()
    print(f"[wiseft] frozen probe = {frozen_probe:.1%} (alpha=0 must match this)", flush=True)

    sweep = []
    for a in args.alphas:
        # Only floating-point tensors are interpolated. Integer buffers such as
        # num_batches_tracked would be corrupted by a weighted average, so they are taken
        # from the fine-tuned state as-is.
        merged = {}
        for k, fv in frozen_visual.items():
            tv = finetuned_visual[k]
            if fv.is_floating_point():
                merged[k] = ((1.0 - a) * fv.float() + a * tv.float()).to(fv.dtype)
            else:
                merged[k] = tv
        model.visual.load_state_dict(merged)
        model.eval()
        s = seen_top1()
        # zeroshot.evaluate() reloads the checkpoint from disk, so it would silently score the
        # ORIGINAL weights and every alpha would return the same unseen number. Score the
        # in-memory interpolated model instead.
        u = _manual_zeroshot(model, tok, preprocess, unseen_rows, unseen_classes,
                             args.strategy, device, torch, F)["acc"]
        sweep.append({"alpha": a, "seen": round(s, 4), "unseen": round(u, 4)})
        print(f"  alpha={a:.2f}  seen={s:.1%}  unseen={u:.1%}", flush=True)

    best = max(sweep, key=lambda r: r["seen"] + r["unseen"])
    out = {
        "tier": args.model, "model": name, "pretrained": pretrained,
        "protocol": f"nested-{args.exp}",
        "seen_classes": len(seen_classes), "seen_images": len(seen_rows),
        "unseen_classes": len(unseen_classes), "unseen_images": len(unseen_rows),
        "unseen_chance": round(1.0 / max(len(unseen_classes), 1), 6),
        "strategy": args.strategy, "ft_epochs": args.epochs, "ft_loss": ft_loss,
        "frozen_probe": round(frozen_probe, 4),
        "sweep": sweep, "best": best,
        "note": ("Both sides measured under ONE protocol. The previous file recorded "
                 "unseen_classes=17 (pilot) against seen_classes=166 (nested C)."),
    }
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = C.RESULTS_DIR / "wiseft.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[wiseft] saved {p}")


def _manual_zeroshot(model, tok, preprocess, rows, classes, strategy, device, torch, F):
    """Zero-shot pass using the CURRENT (interpolated) weights."""
    protos = descriptors.build_prototypes(model, tok, classes, strategy, device)
    cidx = {c: i for i, c in enumerate(classes)}
    correct = tot = 0
    bs = 128
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            from PIL import Image
            x = torch.stack([preprocess(Image.open(r["path"]).convert("RGB")) for r in chunk]).to(device)
            emb = F.normalize(model.encode_image(x), dim=-1)
            pred = (emb @ protos.T).argmax(1).cpu().tolist()
            for r, pi in zip(chunk, pred):
                correct += int(classes[pi] == r["label"]); tot += 1
    return {"acc": correct / max(tot, 1)}


if __name__ == "__main__":
    main()
