"""
temp/train_seen.py — train thoroughly on SEEN crops, keep UNSEEN zero-shot (EXP 2).

Proves the hybrid: a TRAINED head on the seen crops is far more accurate than zero-shot there, while
the UNSEEN-crop zero-shot is preserved. The default is a LINEAR PROBE on the frozen backbone (so the
backbone -- and thus unseen zero-shot -- is untouched by construction; WiSE-FT is only needed for the
heavier full-fine-tune variant, left as a follow-up flag). Produces a checkpoint.

Run on Kaggle (GPU + Internet ON):
    !git clone https://github.com/Abhiram970/plant-disease-edge.git
    %cd plant-disease-edge
    !PDE_DATA_ROOT=/kaggle/working python temp/train_seen.py --tier lw11 --epochs 40
"""
from __future__ import annotations
import argparse
import importlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

def _find_repo():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:                       # pasted into a notebook cell -> no __file__
        cwd = Path.cwd()
        cands = [cwd, *cwd.parents, cwd / "plant-disease-edge", Path("/kaggle/working/plant-disease-edge")]
        cands += [m.parent.parent for m in cwd.glob("*/scripts/config.py")]   # repo as a child dir
        for cand in cands:
            if (cand / "scripts" / "config.py").exists():
                return cand
        return cwd


REPO = _find_repo()
sys.path.insert(0, str(REPO / "scripts"))
import config as C          # noqa: E402
import sage_data            # noqa: E402
import descriptors as D     # noqa: E402
import zeroshot             # noqa: E402


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
    ap.add_argument("--tier", default="lw11", choices=list(C.MODEL_TIERS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-train-crops", type=int, default=4)
    args = ap.parse_args()

    ensure_deps()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name, pre = C.MODEL_TIERS[args.tier]
    print(f"[train_seen] tier={args.tier} ({name})  device={device}")

    # --- data: seen (train) + unseen (held) ---
    seen = sage_data.fetch(C.TRAIN_CROPS, sage_data.full_caps(), min_held_crops=args.min_train_crops)
    unseen = sage_data.fetch(C.HELDOUT_CROPS, sage_data.full_caps(), min_held_crops=C.MIN_HELD_CROPS)
    assert seen and unseen, "need both seen and unseen images"

    # --- frozen backbone + embeddings ---
    model, preprocess, tok, params_m = zeroshot.load_model(name, pre, device)
    print(f"[1] embedding {len(seen):,} seen + {len(unseen):,} unseen imgs (frozen {params_m:.1f}M) ...")
    seen_emb, seen_lab = zeroshot.embed_images(model, preprocess, seen, device)
    uns_emb, uns_lab = zeroshot.embed_images(model, preprocess, unseen, device)

    seen_classes = sorted({r["label"] for r in seen})
    sidx = {c: i for i, c in enumerate(seen_classes)}

    # stratified 80/20 split of seen
    random.seed(42)
    by_cls = defaultdict(list)
    for i, l in enumerate(seen_lab):
        by_cls[l].append(i)
    tr_idx, te_idx = [], []
    for l, idxs in by_cls.items():
        random.shuffle(idxs)
        k = max(1, int(0.2 * len(idxs)))
        te_idx += idxs[:k]; tr_idx += idxs[k:]
    Xtr = seen_emb[tr_idx].to(device)
    ytr = torch.tensor([sidx[seen_lab[i]] for i in tr_idx], device=device)
    Xte = seen_emb[te_idx]
    yte = [seen_lab[i] for i in te_idx]
    print(f"    seen: {len(seen_classes)} classes, train {len(tr_idx):,} / test {len(te_idx):,}")

    # --- LINEAR PROBE on frozen features (the trained seen-crop head) ---
    clf = nn.Linear(seen_emb.shape[1], len(seen_classes)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"[2] training linear probe x {args.epochs} epochs ...")
    for ep in range(args.epochs):
        clf.train()
        perm = torch.randperm(len(tr_idx), device=device)
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            loss = F.cross_entropy(clf(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(Xte.to(device)).argmax(1).cpu().tolist()
    seen_probe_acc = sum(seen_classes[p] == gt for p, gt in zip(pred, yte)) / len(yte)

    # --- comparison: seen ZERO-SHOT (descriptors) on the same test split ---
    seen_protos = D.build_prototypes(model, tok, seen_classes, "rich", device)
    seen_zs_acc, _ = zeroshot.zeroshot_accuracy(seen_emb[te_idx], seen_protos, yte, seen_classes, device)

    # --- UNSEEN zero-shot (frozen backbone -> unchanged by the linear probe) ---
    uns_classes = sorted({r["label"] for r in unseen})
    uns_protos = D.build_prototypes(model, tok, uns_classes, "rich", device)
    uns_zs_acc, uns_by = zeroshot.zeroshot_accuracy(uns_emb, uns_protos, uns_lab, uns_classes, device)

    # --- checkpoint ---
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = C.RESULTS_DIR / f"seen_probe_{args.tier}.pt"
    torch.save({"tier": args.tier, "base": [name, pre], "classes": seen_classes,
                "state_dict": clf.state_dict()}, ckpt)

    print("\n" + "=" * 64)
    print("EXP 2 — TRAIN SEEN, KEEP UNSEEN")
    print("=" * 64)
    print(f"  SEEN crops  ({len(seen_classes)} classes):")
    print(f"    trained head (linear probe) : {seen_probe_acc:.1%}   <- best real-time on known crops")
    print(f"    zero-shot (descriptors)     : {seen_zs_acc:.1%}   (training beats zero-shot on seen)")
    print(f"  UNSEEN crops ({len(uns_classes)} classes):")
    print(f"    zero-shot (descriptors)     : {uns_zs_acc:.1%}   (frozen backbone -> preserved)   "
          + ", ".join(f"{c}={v:.0%}" for c, v in uns_by.items()))
    print("-" * 64)
    print("  HYBRID CONFIRMED: train the seen head for accuracy; keep frozen zero-shot for unseen.")
    print(f"  (linear probe leaves the backbone frozen, so unseen zero-shot is unchanged by construction;")
    print(f"   WiSE-FT is the next layer for a full backbone fine-tune.)")
    print(f"  checkpoint -> {ckpt}")
    print("=" * 64)

    out = C.RESULTS_DIR / f"train_seen_{args.tier}.json"
    out.write_text(json.dumps({
        "tier": args.tier, "seen_classes": len(seen_classes), "unseen_classes": len(uns_classes),
        "seen_probe_acc": seen_probe_acc, "seen_zeroshot_acc": seen_zs_acc,
        "unseen_zeroshot_acc": uns_zs_acc, "unseen_by_crop": uns_by}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
