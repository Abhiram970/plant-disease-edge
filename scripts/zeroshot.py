"""
Zero-shot core — frozen open_clip model + descriptor prototypes -> cross-crop accuracy.

This is the validated method (no training). `evaluate(model_name, pretrained, rows, classes,
strategy)` returns (accuracy, per_crop_dict). Used by scripts/evaluate.py for the model-family x
descriptor-strategy sweep.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import descriptors as D


def load_model(name, pretrained, device="cpu"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    tok = open_clip.get_tokenizer(name)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    img_params_m = sum(p.numel() for p in model.visual.parameters()) / 1e6
    return model, preprocess, tok, img_params_m


def embed_images(model, preprocess, rows, device="cpu"):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    class DS(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            return preprocess(Image.open(rows[i]["path"]).convert("RGB")), rows[i]["label"]

    dl = DataLoader(DS(), batch_size=128, num_workers=2)
    embs, labels = [], []
    with torch.no_grad():
        for imgs, lab in dl:
            embs.append(F.normalize(model.encode_image(imgs.to(device)), dim=-1).cpu())
            labels += list(lab)
    return torch.cat(embs), labels


def zeroshot_accuracy(img_emb, protos, labels, classes, device="cpu"):
    pred = (img_emb.to(device) @ protos.T).argmax(1).cpu().tolist()
    per = defaultdict(lambda: [0, 0]); ok = tot = 0
    for p, gt in zip(pred, labels):
        hit = classes[p] == gt
        ok += hit; tot += 1
        cr = gt.split("|")[0]
        per[cr][0] += hit; per[cr][1] += 1
    return ok / tot, {c: a / n for c, (a, n) in per.items()}


def evaluate(name, pretrained, rows, classes, strategy="rich", device="cpu", reuse_img_emb=None):
    """Eval one model with one descriptor strategy. Pass reuse_img_emb=(emb,labels) to avoid
    re-encoding images across strategies for the same model."""
    model, preprocess, tok, img_params_m = load_model(name, pretrained, device)
    img_emb, labels = reuse_img_emb if reuse_img_emb else embed_images(model, preprocess, rows, device)
    protos = D.build_prototypes(model, tok, classes, strategy, device)
    acc, by_crop = zeroshot_accuracy(img_emb, protos, labels, classes, device)
    return {"img_params_M": round(img_params_m, 2), "acc": acc, "by_crop": by_crop}, (img_emb, labels)
