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
import random
import sys
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
    nw = args.workers
    dl_tr = DataLoader(_SeenDS(tr, tf_tr, cidx), batch_size=args.batch, shuffle=True, num_workers=nw)
    dl_te = DataLoader(_SeenDS(te, tf_te, cidx), batch_size=args.batch, num_workers=nw)

    model = timm.create_model(args.arch, pretrained=True, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[baseline] train {len(tr):,} / test {len(te):,}  x {args.epochs} epochs ...")
    for ep in range(args.epochs):
        model.train(); run = 0.0; nb = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item(); nb += 1
        model.eval(); correct = tot = 0
        with torch.no_grad():
            for x, y in dl_te:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item(); tot += len(y)
        print(f"  epoch {ep+1}/{args.epochs}  loss={run/max(nb,1):.3f}  test_top1={correct/tot:.1%}")

    seen_top1 = correct / tot
    out = {"arch": args.arch, "seen_classes": len(classes), "seen_top1": round(seen_top1, 4),
           "unseen_top1": "structurally 0 (no output neurons for unseen crops; chance at best)",
           "note": "Compare seen_top1 to the frozen-VLM linear probe (~0.67). The CNN CANNOT do "
                   "cross-crop zero-shot — that is the descriptor head's unique capability."}
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = C.RESULTS_DIR / f"supervised_{args.arch}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[baseline] seen top-1 = {seen_top1:.1%}  (unseen: structurally chance)")
    print(f"[baseline] saved {out_path}")


if __name__ == "__main__":
    main()
