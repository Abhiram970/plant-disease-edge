"""
Phase C (rebuttal-proofing) — a conventional supervised CNN baseline on SEEN crops.

Two things a reviewer wants to see:
  (1) the frozen-VLM linear-probe (~67% on seen) is competitive with a normally-trained CNN, and
  (2) a conventional classifier is STRUCTURALLY incapable of unseen-crop diagnosis — it only has
      output neurons for its trained classes, so on an unseen crop it is 0 / chance. This is the
      contrast that makes the descriptor zero-shot head a genuine capability, not a tuning trick.

Trains a small timm CNN (default mobilenetv3_small_100) on the manifest's train_crop classes,
reports test top-1, and states the unseen number (structurally chance). Small + fast on the 4060.

DEPS: timm (already in requirements). USES the manifest from build_manifest.py.

USAGE
  PDE_DATASET_DIR=/c/kaggle/working/exp_data PDE_DATA_ROOT=/c/kaggle/working \
      python scripts/supervised_baseline.py --arch mobilenetv3_small_100 --epochs 8 --batch 64
  python scripts/supervised_baseline.py --arch resnet18 --epochs 10
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys

# MUST precede the torch import: the CUDA allocator reads this once, at initialisation.
# convnextv2_tiny trained three clean epochs at batch 64 and then OOMed at epoch 4 trying to
# allocate 148 MiB with 148 MiB free -- textbook fragmentation, not an oversized batch. Torch's
# own error message recommends exactly this setting, which lets the allocator grow segments
# instead of stranding memory in unusably-sized blocks.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # older torch name
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def read_manifest(role):
    if not C.MANIFEST_CSV.exists():
        sys.exit(f"manifest not found: {C.MANIFEST_CSV} — run scripts/build_manifest.py first.")
    rows = []
    with open(C.MANIFEST_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split_role"] == role:
                rows.append({"path": r["path"], "label": f"{r['crop']}|{r['disease']}"})
    return rows


try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:
    _TorchDataset = object


class _SeenDS(_TorchDataset):
    """Module-level (picklable) dataset so num_workers>0 works under Windows spawn -> ~4x faster."""
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
    ap.add_argument("--arch", default="mobilenetv3_small_100")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-images", type=int, default=15)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (0 if spawn is flaky)")
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    ap.add_argument("--amp", action="store_true",
                    help="Mixed precision + channels_last. ~3-4x faster on T4/P100 and cuts "
                         "activation memory, which is what made batch 128 OOM before.")
    args = ap.parse_args()

    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import torchvision.transforms as T

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_manifest("train_crop")
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    classes = sorted(l for l, rs in by.items() if len(rs) >= args.min_images)
    cidx = {c: i for i, c in enumerate(classes)}
    rows = [r for r in rows if r["label"] in cidx]
    print(f"[baseline] arch={args.arch}  {len(rows):,} seen imgs  {len(classes)} classes  device={device}")

    random.seed(C.RANDOM_SEED)
    tr, te = [], []
    for l in classes:
        idx = by[l][:]
        random.shuffle(idx)
        k = max(1, int(0.2 * len(idx)))
        te += idx[:k]; tr += idx[k:]

    tf_tr = T.Compose([T.Resize((C.IMG_SIZE, C.IMG_SIZE)), T.RandomHorizontalFlip(),
                       T.ToTensor()])
    tf_te = T.Compose([T.Resize((C.IMG_SIZE, C.IMG_SIZE)), T.ToTensor()])

    # _SeenDS is module-level/picklable so workers spawn under Windows; keep it modest for stability.
    # Kaggle gives 4 vCPUs. The previous settings (plain workers, no prefetch, no pinning, no
    # persistence) starved the GPU: ~1400 s/epoch, i.e. 23 minutes, which made 14 architectures
    # impossible inside a 12 h session. Persistent workers avoid re-forking every epoch, prefetch
    # keeps the queue full, and pinned memory makes the H2D copy async.
    nw = args.workers
    _kw = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        _kw.update(persistent_workers=True, prefetch_factor=4)
    dl_tr = DataLoader(_SeenDS(tr, tf_tr, cidx), batch_size=args.batch, shuffle=True,
                       drop_last=True, **_kw)
    dl_te = DataLoader(_SeenDS(te, tf_te, cidx), batch_size=args.batch, **_kw)

    model = timm.create_model(args.arch, pretrained=True, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # --- Precision selection -------------------------------------------------------------
    # `torch.cuda.is_bf16_supported()` returns True on Turing (T4, sm_75), where bf16 is
    # emulated rather than native. cuDNN then has no convolution engine for bf16 depthwise
    # convs and raises "GET was unable to find an engine to execute this computation" -- which
    # is what killed all ten depthwise architectures (MobileNet, EfficientNet, FastViT,
    # ConvNeXt-V2) in the 2026-09-06 run while the four plain-conv nets trained fine.
    # bf16 is only requested where it is native: Ampere (sm_80) and later.
    _use_amp = bool(args.amp and device == "cuda")
    _cc = torch.cuda.get_device_capability() if device == "cuda" else (0, 0)
    _bf16_native = _cc[0] >= 8
    _plan = []
    if _use_amp:
        if _bf16_native:
            _plan.append(("bf16", torch.bfloat16, True))
        _plan.append(("fp16", torch.float16, True))       # T4 tensor cores are fp16
        _plan.append(("fp16-contig", torch.float16, False))
        _plan.append(("fp32", torch.float32, False))
    else:
        _plan.append(("fp32", torch.float32, False))

    _ENGINE_ERR = "unable to find an engine"

    def _apply(mode):
        """Install one precision plan; returns (dtype, channels_last, scaler)."""
        _name, _dt, _cl = mode
        model.to(memory_format=torch.channels_last if _cl else torch.contiguous_format)
        _sc = torch.amp.GradScaler("cuda", enabled=(_dt is torch.float16))
        print(f"[baseline] precision={_name}"
              f"{' + channels_last' if _cl else ''} (sm_{_cc[0]}{_cc[1]})", flush=True)
        return _dt, _cl, _sc

    _plan_i = 0
    _adt, _chlast, scaler = _apply(_plan[0])

    # --- Checkpointing ---
    ckpt_dir = C.DATA_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.arch}_ckpt.pt"
    start_epoch = 0
    best_acc = 0.0
    epoch_log = []  # list of {epoch, loss, test_top1}

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        epoch_log = ckpt.get("epoch_log", [])
        print(f"[baseline] RESUMED from epoch {start_epoch} (best={best_acc:.1%})")
    # ----------------------

    print(f"[baseline] train {len(tr):,} / test {len(te):,}  x {args.epochs} epochs ...")
    # Initialised here, not in the loop: resuming a run whose epochs are all complete skips the loop
    # body entirely, and the summary below would otherwise raise NameError on `correct`/`tot`.
    correct = tot = 0
    for ep in range(start_epoch, args.epochs):
        model.train(); run = 0.0; nb = 0
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if _chlast:
                x = x.to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            try:
                with torch.autocast("cuda", dtype=_adt,
                                    enabled=(_adt is not torch.float32)):
                    loss = F.cross_entropy(model(x), y)
                if scaler.is_enabled():
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
            except RuntimeError as e:
                # cuDNN has no kernel for this (dtype, memory-format, conv) combination.
                # Step down the ladder and retry the same batch rather than losing the arch.
                if _ENGINE_ERR not in str(e) or _plan_i + 1 >= len(_plan):
                    raise
                _plan_i += 1
                print(f"[baseline] cuDNN found no engine for {_plan[_plan_i - 1][0]}; "
                      f"falling back to {_plan[_plan_i][0]}", flush=True)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                _adt, _chlast, scaler = _apply(_plan[_plan_i])
                x = x.to(memory_format=torch.channels_last if _chlast
                         else torch.contiguous_format)
                with torch.autocast("cuda", dtype=_adt,
                                    enabled=(_adt is not torch.float32)):
                    loss = F.cross_entropy(model(x), y)
                if scaler.is_enabled():
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
            run += loss.item(); nb += 1
        model.eval(); correct = tot = 0
        with torch.no_grad():
            for x, y in dl_te:
                x = x.to(device, non_blocking=True)
                if _chlast:
                    x = x.to(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=_adt,
                                    enabled=(_adt is not torch.float32)):
                    pred = model(x).argmax(1).cpu()
                correct += (pred == y).sum().item(); tot += len(y)
        acc = correct/tot
        print(f"  epoch {ep+1}/{args.epochs}  loss={run/max(nb,1):.3f}  test_top1={acc:.1%}")
        # Release the eval graph and return cached blocks before checkpointing. convnextv2_tiny
        # OOMed at epoch 4 having trained three epochs cleanly at the same batch size, which is
        # fragmentation accumulating across epochs, not a batch that never fit. torch.save also
        # briefly holds a CPU copy of every optimizer tensor, so the peak lands right here.
        # Bound to None rather than `del`: a bare `del x, y, pred` raises NameError when the eval
        # loader yielded nothing, and locals().pop() only mutates a throwaway dict inside a
        # function. Rebinding drops the last reference just as effectively.
        x = y = pred = loss = None
        torch.cuda.empty_cache()

        # Save checkpoint
        epoch_log.append({"epoch": ep, "loss": round(run/max(nb,1), 4), "test_top1": round(acc, 4)})
        if acc > best_acc:
            best_acc = acc
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "epoch": ep, "best_acc": best_acc, "epoch_log": epoch_log,
                     "classes": classes}, ckpt_path)
        print(f"    [ckpt] saved epoch {ep+1} (best={best_acc:.1%})")

    seen_top1 = correct / tot if tot > 0 else best_acc
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    out = {"arch": args.arch, "seen_classes": len(classes), "seen_top1": round(seen_top1, 4),
           "params_M": round(n_params, 2), "epochs": args.epochs, "batch": args.batch,
           "img_size": C.IMG_SIZE, "epoch_log": epoch_log,
           "unseen_top1": "structurally 0 (no output neurons for unseen crops; chance at best)",
           "note": "Compare seen_top1 to the frozen-VLM linear probe (0.824 on the same 166 "
                   "classes). The CNN CANNOT do cross-crop zero-shot — that is the descriptor "
                   "head's unique capability."}
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = C.RESULTS_DIR / f"supervised_{args.arch}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[baseline] seen top-1 = {seen_top1:.1%}  (unseen: structurally chance)")
    print(f"[baseline] saved {out_path}")


if __name__ == "__main__":
    main()
