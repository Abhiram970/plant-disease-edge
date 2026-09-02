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
    ap.add_argument("--extract-workers", type=int, default=2,
                    help="Workers for the feature-extraction passes. Separate from --workers: "
                         "the crash that forced --workers 0 was in the TRAINING loader, which "
                         "is built while a CUDA context and a live model already exist. The "
                         "extraction loaders can safely use workers, and they dominate the "
                         "runtime (every alpha re-extracts).")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers. Defaults to 0: unlike the CNN baseline, this "
                         "script has a CUDA context and a loaded model live BEFORE the loader "
                         "is built, and spawning workers around that crashed them outright "
                         "('DataLoader worker exited unexpectedly'). The dataset here is the "
                         "seen split for a few epochs, so the loader is not the bottleneck.")
    ap.add_argument("--strategy", default="grounded")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--min-images", type=int, default=15)
    ap.add_argument("--max-per-class", type=int, default=200,
                    help="Cap seen images per class for the sweep. The alpha sweep measures a "
                         "TREND across alpha, not an absolute leaderboard number, and every "
                         "alpha re-extracts features over the whole split: at 70k images and "
                         "3 alphas that is ~1 h of JPEG decoding before any training. A "
                         "stratified cap keeps the trend and makes the stage affordable. "
                         "Pass 0 to use every image.")
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
    if args.max_per_class:
        _before = len(seen_rows)
        _rng = random.Random(C.RANDOM_SEED)
        _capped = []
        for _l in seen_classes:
            _rs = by[_l][:]
            _rng.shuffle(_rs)
            _capped += _rs[:args.max_per_class]
        seen_rows = _capped
        by = defaultdict(list)
        for r in seen_rows:
            by[r["label"]].append(r)
        print(f"[wiseft] seen split capped at {args.max_per_class}/class: "
              f"{_before:,} -> {len(seen_rows):,} images", flush=True)
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

    # Training loaders: --workers (default 0). Spawning workers around a live CUDA context
    # and a loaded model killed them outright, which is why this defaults to single-process.
    _kw = dict(num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        _kw.update(persistent_workers=True, prefetch_factor=4)
    dl_tr = DataLoader(_WiseDS(tr, preprocess, cidx), batch_size=args.batch, shuffle=True,
                       drop_last=True, **_kw)
    dl_te = DataLoader(_WiseDS(te, preprocess, cidx), batch_size=args.batch, **_kw)

    # Extraction loaders: separate, and allowed workers. Every alpha re-extracts features over
    # the whole seen split, so this is where the runtime actually goes -- at workers=0 the
    # sweep is roughly an hour of single-threaded JPEG decoding before any training happens.
    _ekw = dict(num_workers=args.extract_workers, pin_memory=True)
    if args.extract_workers > 0:
        _ekw.update(persistent_workers=True, prefetch_factor=4)
    ex_tr = DataLoader(_WiseDS(tr, preprocess, cidx), batch_size=args.batch, **_ekw)
    ex_te = DataLoader(_WiseDS(te, preprocess, cidx), batch_size=args.batch, **_ekw)

    # ---- fine-tune the VISUAL tower on seen crops (linear head on top, both trained) ------
    with torch.no_grad():
        dim = model.visual(next(iter(dl_te))[0][:1].to(device)).shape[-1]
    head = nn.Linear(dim, len(seen_classes)).to(device)
    # zeroshot.load_model() sets requires_grad=False on EVERY parameter -- correct for its own
    # job (frozen zero-shot eval) but fatal here: the optimizer held visual parameters that
    # could not receive gradients, so the "fine-tuned" encoder was byte-identical to the frozen
    # one and only the head moved. That is why raising the lr from 1e-5 to 1e-4 barely shifted
    # the loss (5.078 -> 4.803 against a random-guess 5.111) and why alpha=0.5 dipped below both
    # endpoints: there was no second weight set to interpolate toward. Re-enable grads on the
    # visual tower, which is the thing WiSE-FT interpolates.
    for _p in model.visual.parameters():
        _p.requires_grad = True
    _trainable = sum(_p.numel() for _p in model.visual.parameters() if _p.requires_grad)
    print(f"[wiseft] visual tower trainable params: {_trainable/1e6:.2f} M", flush=True)
    if _trainable == 0:
        sys.exit("[wiseft] visual tower has no trainable parameters; refusing to run.")
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

    # Did the encoder actually move? A frozen requires_grad flag made the "fine-tuned" weights
    # byte-identical to the frozen ones once already, and every downstream number still looked
    # plausible -- alpha=1 differed only because the head differed. Measure the distance and
    # refuse to continue if it is zero.
    _delta = 0.0
    _norm = 0.0
    for _k, _fv in frozen_visual.items():
        if _fv.is_floating_point():
            _delta += (finetuned_visual[_k].float() - _fv.float()).pow(2).sum().item()
            _norm += _fv.float().pow(2).sum().item()
    _rel = (_delta ** 0.5) / max(_norm ** 0.5, 1e-12)
    print(f"[wiseft] encoder moved: relative L2 distance = {_rel:.5f}", flush=True)
    if _rel < 1e-8:
        sys.exit("[wiseft] the fine-tuned encoder is identical to the frozen one -- nothing was "
                 "trained. Refusing to emit a sweep that would interpolate a model with itself.")

    # ---- sweep alpha ----------------------------------------------------------------------
    # The alpha=0 row must reproduce the frozen probe -- that is the sweep's built-in
    # correctness check -- so the frozen probe is measured here as the reference. Two smoke
    # tests shaped this: training the head jointly with the encoder gave seen=3.3% at alpha=0
    # (the head, not the encoder, was untrained), and then holding one frozen-fitted head
    # across the whole sweep gave seen=26.3% at alpha=1, because a fine-tuned encoder moves the
    # feature space out from under a stale head. Each alpha therefore gets its own head, fitted
    # to that encoder, so the seen column reflects the ENCODER interpolation and nothing else.
    print("[wiseft] fitting the frozen-encoder head (the alpha=0 reference) ...", flush=True)
    model.visual.load_state_dict(frozen_visual)
    frozen_probe = _fit_and_score(model, ex_tr, ex_te, len(seen_classes), device, _adt,
                                  torch, nn, F)
    print(f"[wiseft] frozen probe = {frozen_probe:.1%} (alpha=0 must reproduce this)", flush=True)

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
        # Re-fit the linear head on THIS encoder's features before scoring the seen side.
        # Holding the frozen-fitted head fixed made alpha=1 report 26.3% seen in a smoke test:
        # the fine-tuned encoder moves the feature space out from under a stale head, so the
        # number measured head staleness rather than the encoder. WiSE-FT's actual claim is
        # that interpolation BUYS seen accuracy, which requires each alpha to be scored with a
        # head fitted to it. Re-fitting is cheap -- features are extracted once per alpha and
        # the head is a single linear layer.
        s = _fit_and_score(model, ex_tr, ex_te, len(seen_classes), device, _adt, torch, nn, F)
        # zeroshot.evaluate() reloads the checkpoint from disk, so it would silently score the
        # ORIGINAL weights and every alpha would return the same unseen number. Score the
        # in-memory interpolated model instead.
        u = _manual_zeroshot(model, tok, preprocess, unseen_rows, unseen_classes,
                             args.strategy, device, torch, F)["acc"]
        sweep.append({"alpha": a, "seen": round(s, 4), "unseen": round(u, 4)})
        print(f"  alpha={a:.2f}  seen={s:.1%}  unseen={u:.1%}", flush=True)

    # ---- sanity gates -------------------------------------------------------------------
    # WiSE-FT only behaves when the fine-tuned model is genuinely fine-tuned from the same
    # init (Wortsman et al.). If fine-tuning barely moved, the frozen and "fine-tuned" weights
    # are not on a connected low-loss path and the midpoint lands in a degenerate region --
    # in a 1-epoch smoke test that produced seen = 72.3 / 63.9 / 73.6, a dip BELOW both
    # endpoints. That is a training-budget failure, not a result, so say so loudly rather than
    # emitting a table someone might read as the seen/unseen trade-off.
    import math as _math
    random_loss = _math.log(max(len(seen_classes), 2))
    warnings = []
    if ft_loss and ft_loss[-1] > 0.9 * random_loss:
        warnings.append(
            f"fine-tuning did not converge: final loss {ft_loss[-1]:.2f} vs "
            f"random-guess {random_loss:.2f}. Raise --epochs or --lr; the sweep is not "
            f"interpretable until alpha=1 is a genuinely fine-tuned model.")
    seen_curve = [r["seen"] for r in sweep]
    if len(seen_curve) >= 3 and min(seen_curve) < min(seen_curve[0], seen_curve[-1]) - 0.01:
        warnings.append(
            "seen accuracy dips below BOTH endpoints at an intermediate alpha, which means "
            "the two weight sets are not linearly connected -- usually the same "
            "under-training cause as above.")
    a0 = next((r for r in sweep if r["alpha"] == 0.0), None)
    if a0 and abs(a0["seen"] - frozen_probe) > 0.01:
        warnings.append(
            f"alpha=0 ({a0['seen']:.1%}) does not reproduce the frozen probe "
            f"({frozen_probe:.1%}); the interpolation itself is suspect.")
    for w in warnings:
        print(f"[wiseft][WARNING] {w}", flush=True)

    best = max(sweep, key=lambda r: r["seen"] + r["unseen"])
    out = {
        "tier": args.model, "model": name, "pretrained": pretrained,
        "protocol": f"nested-{args.exp}",
        "seen_classes": len(seen_classes), "seen_images": len(seen_rows),
        "unseen_classes": len(unseen_classes), "unseen_images": len(unseen_rows),
        "unseen_chance": round(1.0 / max(len(unseen_classes), 1), 6),
        "strategy": args.strategy, "ft_epochs": args.epochs, "ft_loss": ft_loss,
        "frozen_probe": round(frozen_probe, 4),
        "encoder_rel_l2_shift": round(_rel, 6),
        "random_guess_loss": round(random_loss, 4),
        "warnings": warnings,
        "sweep": sweep, "best": best,
        "note": ("Both sides measured under ONE protocol. The previous file recorded "
                 "unseen_classes=17 (pilot) against seen_classes=166 (nested C)."),
    }
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = C.RESULTS_DIR / "wiseft.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[wiseft] saved {p}")


def _fit_and_score(model, ex_tr, ex_te, n_classes, device, adt, torch, nn, F, epochs=20):
    """Fit a linear head on the CURRENT encoder's frozen features and return test top-1.

    Used for every alpha so the seen column reflects the interpolated ENCODER, not how stale
    a fixed head has become. Returns accuracy in [0, 1].
    """
    model.eval()
    with torch.no_grad():
        Xtr, Ytr, Xte, Yte = [], [], [], []
        for dl, X, Y in ((ex_tr, Xtr, Ytr), (ex_te, Xte, Yte)):
            for x, y in dl:
                x = x.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=adt, enabled=(device == "cuda")):
                    X.append(model.visual(x).float().cpu())
                Y.append(y)
        Xtr = torch.cat(Xtr); Ytr = torch.cat(Ytr)
        Xte = torch.cat(Xte); Yte = torch.cat(Yte)
    head = nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(head(Xtr[b].to(device)), Ytr[b].to(device)).backward()
            opt.step()
    with torch.no_grad():
        return (head(Xte.to(device)).argmax(1).cpu() == Yte).float().mean().item()


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
