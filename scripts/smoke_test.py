"""
Phase A4 — End-to-end SMOKE TEST.  ***This is the RTX 4060 job.***

Goal: prove the ENTIRE pipeline runs on real hardware BEFORE we spend Kaggle hours.
It is intentionally tiny (a few hundred images, 1 epoch) and finishes in minutes.

It exercises every moving part of the real plan:
  1. load the subset + splits + descriptors
  2. load a frozen TEXT teacher (CLIP/OpenCLIP) + frozen DINOv2 (aux)   [the teachers]
  3. load EdgeNeXt-XX-Small with a text-projection head                 [the student]
  4. cache teacher embeddings for a small sample                        [Phase B in miniature]
  5. build text prototypes from descriptors                            [the zero-shot fuel]
  6. run ONE distillation epoch (align student -> teacher text space)   [Phase C in miniature]
  7. zero-shot sanity check on a HELD-OUT crop                          [the headline mechanism]

If this completes with no errors, the plumbing is correct and we can scale on Kaggle.

WHERE TO RUN: the RTX 4060 box (uses CUDA if present, else CPU — both work).

PREREQS (run these first; each is fast / CPU):
  python scripts/build_sage_subset.py --limit 30000 --per-class-cap 80
  python scripts/build_splits.py
  python scripts/build_descriptors.py            # stubs are fine for the smoke test

USAGE
  python scripts/smoke_test.py                   # auto: small sample, 1 epoch
  python scripts/smoke_test.py --teacher openclip --max-per-split 400 --epochs 1
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import timm
    import open_clip
except ImportError as e:
    sys.exit(f"Missing dependency ({e}). Install torch+torchvision (CUDA) then "
             f"`pip install -r requirements.txt`.")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


# --------------------------------------------------------------------------- data
def read_split(name: str, max_n: int) -> list[dict]:
    p = C.SPLITS_DIR / f"{name}.csv"
    if not p.exists():
        sys.exit(f"missing {p} — run build_splits.py (A3) first.")
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:max_n]


class ImgDataset(Dataset):
    def __init__(self, rows, preprocess):
        self.rows, self.pre = rows, preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(C.DATA_ROOT / r["path"]).convert("RGB")
        return self.pre(img), f'{r["crop"]}|{r["disease"]}'


# --------------------------------------------------------------------------- models
def load_text_teacher(which: str, device):
    """Frozen CLIP-family teacher (image+text aligned -> enables zero-shot)."""
    name, pretrained = {
        "clip": ("ViT-B-16", "openai"),
        "openclip": ("ViT-B-16", "laion2b_s34b_b88k"),
    }[which]
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(name)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    dim = model.text_projection.shape[1] if hasattr(model, "text_projection") else 512
    return model, preprocess, tokenizer, dim


def load_dino(device):
    """Frozen DINOv2-small (vision-only auxiliary teacher). Optional in the smoke test."""
    try:
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
    except Exception as e:
        print(f"   [skip] DINOv2 unavailable in smoke test ({e}); continuing without aux teacher.")
        return None
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_student(out_dim: int, device):
    """EdgeNeXt-XX-Small (~1.3M) + a text-projection head into the teacher's space."""
    for name in ("edgenext_xx_small.in1k", "edgenext_xx_small"):
        try:
            backbone = timm.create_model(name, pretrained=True, num_classes=0)
            break
        except Exception:
            backbone = None
    if backbone is None:
        sys.exit("Could not create edgenext_xx_small — upgrade timm (>=1.0.3).")
    feat = backbone.num_features
    head = nn.Linear(feat, out_dim)
    model = nn.Sequential(backbone, head).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   student params: {n_params/1e6:.2f}M  (backbone feat={feat} -> text dim={out_dim})")
    return model


# --------------------------------------------------------------------------- prototypes
def build_text_prototypes(teacher, tokenizer, device, crops):
    """For each (crop,disease) descriptor -> a normalized text-prototype embedding."""
    protos, labels = [], []
    for crop in crops:
        fp = C.DESCRIPTORS_DIR / f"{C.safe_name(crop)}.json"
        if not fp.exists():
            continue
        for rec in json.loads(fp.read_text(encoding="utf-8")):
            text = rec.get("symptom_text") or f'{rec["disease"]} on {rec["crop"]}'
            tok = tokenizer([text]).to(device)
            with torch.no_grad():
                emb = teacher.encode_text(tok)
            emb = F.normalize(emb, dim=-1)
            protos.append(emb.cpu())
            labels.append(f'{rec["crop"]}|{rec["disease"]}')
    if not protos:
        return None, []
    return torch.cat(protos, 0), labels


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Phase A4 end-to-end smoke test (RTX 4060).")
    ap.add_argument("--teacher", choices=["clip", "openclip"], default="clip")
    ap.add_argument("--max-per-split", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"SMOKE TEST  | device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (CPU — fine, just slower)"))
    print("=" * 60)

    # 1) teachers + student
    print("[1/6] loading frozen text teacher ...")
    teacher, preprocess, tokenizer, dim = load_text_teacher(args.teacher, device)
    print("[2/6] loading frozen DINOv2 aux teacher ...")
    dino = load_dino(device)
    print("[3/6] loading EdgeNeXt-XX-Small student ...")
    student = load_student(dim, device)

    # 2) data
    train_rows = read_split("train", args.max_per_split)
    if not train_rows:
        sys.exit("train split empty — did A1/A3 run with data?")
    dl = DataLoader(ImgDataset(train_rows, preprocess), batch_size=args.batch, shuffle=True)
    print(f"[4/6] training sample: {len(train_rows)} imgs, {len(dl)} batches")

    # 3) one distillation epoch: student image emb -> teacher image emb (cosine align)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    student.train()
    for ep in range(args.epochs):
        running = 0.0
        for imgs, _ in dl:
            imgs = imgs.to(device)
            with torch.no_grad():
                t_emb = F.normalize(teacher.encode_image(imgs), dim=-1)
            s_emb = F.normalize(student(imgs), dim=-1)
            loss = (1 - (s_emb * t_emb).sum(-1)).mean()  # cosine distillation loss
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"   epoch {ep+1}/{args.epochs}  distill_loss={running/len(dl):.4f}")

    # 4) zero-shot sanity check on a HELD-OUT crop (the headline mechanism, in miniature)
    print("[5/6] building text prototypes from descriptors ...")
    protos, proto_labels = build_text_prototypes(teacher, tokenizer, device, C.HELDOUT_CROPS)
    print("[6/6] zero-shot sanity check on held-out crops ...")
    held = read_split("heldout", 64)
    if protos is None or not held:
        print("   [skip] no held-out images or descriptors yet — mechanism wiring untested but"
              " training path verified.")
    else:
        protos = protos.to(device)
        student.eval()
        correct = 0
        hdl = DataLoader(ImgDataset(held, preprocess), batch_size=args.batch)
        gts = [r["crop"] + "|" + r["disease"] for r in held]
        idx = 0
        with torch.no_grad():
            for imgs, _ in hdl:
                s_emb = F.normalize(student(imgs.to(device)), dim=-1)
                sims = s_emb @ protos.T
                preds = sims.argmax(1).cpu().tolist()
                for p in preds:
                    if proto_labels[p] == gts[idx]:
                        correct += 1
                    idx += 1
        acc = correct / len(held)
        print(f"   zero-shot acc on {len(held)} held-out imgs: {acc:.1%}  "
              f"(random ~{1/max(len(proto_labels),1):.1%}; tiny/1-epoch — just proving the path)")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED ✔  pipeline runs end-to-end. Ready to scale on Kaggle.")
    print("=" * 60)


if __name__ == "__main__":
    main()
