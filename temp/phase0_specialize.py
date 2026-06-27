"""
Phase 0 SPECIALIZE — does agricultural specialization LIFT a pretrained small CLIP's
cross-crop zero-shot?   (THE load-bearing experiment for the headline)
=====================================================================================
The probe showed a frozen MobileCLIP2-S0 (~11M) already does cross-crop disease
zero-shot (~20% on 17 classes, 3.4x chance, ~78% of the 93M SigLIP2 teacher). The
paper's contribution is the LIFT we add by specializing it on agriculture.

RISK we design around = catastrophic forgetting: fine-tuning hard on TRAIN crops can
destroy the general image-text alignment that gives UNSEEN-crop zero-shot. So we do
forgetting-resistant, parameter-efficient specialization:
  - base image + text encoders FROZEN (pretrained alignment preserved);
  - a small RESIDUAL ADAPTER on the image embedding:  z' = (1-a)*z + a*MLP(z)  (CLIP-Adapter);
  - trained on TRAIN crops with: CE(image -> its descriptor prototype) + KD(distill the
    SigLIP2 teacher's class distribution);
  - frozen base embeddings are CACHED once -> the adapter trains in minutes.

We then measure held-out (Coffee/Orange/Peach) zero-shot BEFORE vs AFTER. The LIFT is
the result:
  lift >= +5pp -> GO (specialization works -> full Phase A-E).
  ~0 or negative -> headline pivots to efficiency + descriptor-FM story (still a paper).

Reuses the on-disk SAGE images from phase0_spike.py.
Run on Kaggle (GPU, Internet ON):  %run phase0_specialize.py
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("/kaggle/working/spike_data")
if not DATA_DIR.exists():
    DATA_DIR = Path("./phase0_out/spike_data")
TRAIN_CROPS = ["Tomato", "Apple", "Corn", "Grape", "Potato"]   # whatever is on disk
HELDOUT_CROPS = ["Coffee", "Orange", "Peach"]
MIN_CLASS_IMAGES = 15
RESULT_JSON = Path("/kaggle/working/phase0_specialize_result.json")

STUDENT = ("MobileCLIP2-S0", "dfndr2b")    # ~11M image encoder = the small tier
TEACHER = ("ViT-B-16-SigLIP2", "webli")    # best probe teacher (25.6%)
ALPHA      = 0.3      # residual adapter mix (small -> forgetting-resistant)
TAU        = 0.07
LAMBDA_KD  = 1.0      # weight of SigLIP2 distillation vs CE-to-descriptors
EPOCHS     = 50
BATCH      = 256
LR         = 1e-3
SEED       = 42

SYMPTOM_HINTS = {
    "rust": "orange to brown powdery pustules on the underside of the leaf",
    "blight": "rapidly spreading brown necrotic lesions and dead leaf tissue",
    "spot": "small dark circular spots with concentric rings on the leaf",
    "mildew": "a white or grey powdery fungal coating on the leaf surface",
    "canker": "sunken corky lesions with yellow halos on the leaf",
    "greening": "blotchy asymmetric yellow mottling of the leaf",
    "huanglongbing": "blotchy asymmetric yellow mottling of the leaf",
    "curl": "puckered, thickened, distorted and reddened curled leaves",
    "brown rot": "brown spreading rot with tan fungal spore masses",
    "scab": "olive-green to black velvety scabby lesions on the leaf",
    "mosaic": "a mottled light-and-dark green mosaic pattern on the leaf",
    "cercospora": "brown spots with grey centers and yellow halos on the leaf",
    "deficiency": "interveinal yellowing of the leaf from nutrient deficiency",
    "healthy": "a healthy green leaf with no disease symptoms",
}
TEMPLATES = ["a photo of {}", "a close-up leaf photo: {}", "a leaf with {}"]


def descriptor_text(lbl):
    crop, dis = lbl.split("|")
    k = dis.lower()
    hint = next((v for kw, v in SYMPTOM_HINTS.items() if kw in k), "")
    base = f"{dis} on {crop} leaf".replace("_", " ")
    return f"{base}: {hint}" if hint else base


def ensure_deps():
    import importlib, subprocess
    need = []
    for mod, pkg in [("open_clip", "open_clip_torch>=2.24"), ("timm", "timm>=1.0.3")]:
        try:
            importlib.import_module(mod)
        except Exception:
            need.append(pkg)
    if need:
        print(f"[0] installing missing deps: {need}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)


def load_rows(crops, min_imgs=MIN_CLASS_IMAGES):
    rows = []
    if not DATA_DIR.exists():
        sys.exit(f"no data dir {DATA_DIR}; run phase0_spike.py first.")
    for d in DATA_DIR.iterdir():
        if not d.is_dir() or "___" not in d.name:
            continue
        crop, disease = d.name.split("___", 1)
        if crop not in crops:
            continue
        for jpg in d.glob("*.jpg"):
            rows.append({"path": str(jpg), "label": f"{crop}|{disease}"})
    cc = Counter(r["label"] for r in rows)
    return [r for r in rows if cc[r["label"]] >= min_imgs]


def main():
    ensure_deps()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import open_clip
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] device={dev}")

    (sname, spre), (tname, tpre) = STUDENT, TEACHER
    student, _, s_pre = open_clip.create_model_and_transforms(sname, pretrained=spre)
    s_tok = open_clip.get_tokenizer(sname)
    teacher, _, t_pre = open_clip.create_model_and_transforms(tname, pretrained=tpre)
    t_tok = open_clip.get_tokenizer(tname)
    student.eval().to(dev)
    teacher.eval().to(dev)
    for p in student.parameters():
        p.requires_grad = False
    for p in teacher.parameters():
        p.requires_grad = False

    train_rows = load_rows(TRAIN_CROPS)
    held_rows = load_rows(HELDOUT_CROPS)
    assert train_rows and held_rows, "need both train and held images on disk"
    train_classes = sorted({r["label"] for r in train_rows})
    held_classes = sorted({r["label"] for r in held_rows})
    tr_idx = {c: i for i, c in enumerate(train_classes)}
    chance = 1.0 / len(held_classes)
    print(f"[*] train: {len(train_rows):,} imgs / {len(train_classes)} classes "
          f"({sorted({c.split('|')[0] for c in train_classes})})")
    print(f"[*] held : {len(held_rows):,} imgs / {len(held_classes)} classes  chance={chance:.1%}")

    def build_protos(model, tok, classes):
        out = []
        with torch.no_grad():
            for lbl in classes:
                toks = tok([t.format(descriptor_text(lbl)) for t in TEMPLATES]).to(dev)
                emb = F.normalize(model.encode_text(toks), dim=-1).mean(0)
                out.append(F.normalize(emb, dim=-1))
        return torch.stack(out).to(dev)

    s_tr_protos = build_protos(student, s_tok, train_classes)
    s_he_protos = build_protos(student, s_tok, held_classes)
    t_tr_protos = build_protos(teacher, t_tok, train_classes)
    t_he_protos = build_protos(teacher, t_tok, held_classes)

    class DS(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            img = Image.open(self.rows[i]["path"]).convert("RGB")
            return s_pre(img), t_pre(img), self.rows[i]["label"]

    def embed(rows):
        dl = DataLoader(DS(rows), batch_size=128, num_workers=2)
        S, T, L = [], [], []
        with torch.no_grad():
            for si, ti, lab in dl:
                S.append(F.normalize(student.encode_image(si.to(dev)), dim=-1).cpu())
                T.append(F.normalize(teacher.encode_image(ti.to(dev)), dim=-1).cpu())
                L += list(lab)
        return torch.cat(S), torch.cat(T), L

    print("[1] precomputing frozen embeddings (one pass over images) ...")
    Str, Ttr, Ltr = embed(train_rows)
    She, The, Lhe = embed(held_rows)

    def zshot(emb, protos, labels, classes):
        pred = (emb.to(dev) @ protos.T).argmax(1).cpu().tolist()
        per = defaultdict(lambda: [0, 0]); ok = tot = 0
        for p, gt in zip(pred, labels):
            hit = classes[p] == gt; ok += hit; tot += 1
            cr = gt.split("|")[0]; per[cr][0] += hit; per[cr][1] += 1
        return ok / tot, {c: a / n for c, (a, n) in per.items()}

    base_acc, base_by = zshot(She, s_he_protos, Lhe, held_classes)
    teach_acc, _ = zshot(The, t_he_protos, Lhe, held_classes)
    print(f"[2] BEFORE  student frozen held zero-shot = {base_acc:.1%}   (SigLIP2 teacher {teach_acc:.1%})")

    d = Str.shape[1]

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, d))
        def forward(self, z):
            return F.normalize((1 - ALPHA) * z + ALPHA * self.net(z), dim=-1)

    adapter = Adapter().to(dev)
    opt = torch.optim.AdamW(adapter.parameters(), lr=LR)
    y = torch.tensor([tr_idx[l] for l in Ltr], device=dev)
    Str_d = Str.to(dev)
    with torch.no_grad():
        t_tr_logits = (Ttr.to(dev) @ t_tr_protos.T) / TAU      # frozen teacher class structure

    nparams = sum(p.numel() for p in adapter.parameters())
    print(f"[3] training adapter ({nparams/1e3:.0f}K params, alpha={ALPHA}, lambda_kd={LAMBDA_KD}) "
          f"x {EPOCHS} epochs on cached embeddings ...")
    N = Str_d.shape[0]
    best = base_acc
    for ep in range(EPOCHS):
        adapter.train()
        perm = torch.randperm(N, device=dev)
        rc = rk = 0.0; nb = 0
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            z = adapter(Str_d[idx])
            s_logits = (z @ s_tr_protos.T) / TAU
            ce = F.cross_entropy(s_logits, y[idx])
            kd = F.kl_div(F.log_softmax(s_logits, 1), F.softmax(t_tr_logits[idx], 1),
                          reduction="batchmean")
            loss = ce + LAMBDA_KD * kd
            opt.zero_grad(); loss.backward(); opt.step()
            rc += ce.item(); rk += kd.item(); nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            adapter.eval()
            with torch.no_grad():
                acc, _ = zshot(adapter(She.to(dev)).cpu(), s_he_protos, Lhe, held_classes)
            best = max(best, acc)
            print(f"    epoch {ep+1:3d}/{EPOCHS}  ce={rc/nb:.3f}  kd={rk/nb:.3f}  held_zshot={acc:.1%}")

    adapter.eval()
    with torch.no_grad():
        spec_acc, spec_by = zshot(adapter(She.to(dev)).cpu(), s_he_protos, Lhe, held_classes)
    lift = spec_acc - base_acc

    print("\n" + "=" * 64)
    print("PHASE 0 SPECIALIZE — RESULT")
    print("=" * 64)
    print(f"  held-out crops          : {HELDOUT_CROPS}  ({len(held_classes)} classes, "
          f"{len(held_rows):,} imgs, chance {chance:.1%})")
    print(f"  student base  (frozen)  : {base_acc:.1%}")
    print(f"  student SPECIALIZED     : {spec_acc:.1%}   (lift {lift:+.1%};  best-epoch {best:.1%})")
    print(f"  teacher SigLIP2 (ref)   : {teach_acc:.1%}")
    print(f"  per-crop (specialized)  : "
          + ", ".join(f"{c}={spec_by.get(c, 0):.0%}" for c in HELDOUT_CROPS if c in spec_by))
    print("-" * 64)
    if lift >= 0.05:
        verdict = "GO — specialization lifts cross-crop zero-shot -> proceed to full Phase A-E."
    elif lift > 0.0:
        verdict = "WEAK — small lift; try real source-grounded descriptors / tune alpha,lambda before deciding."
    else:
        verdict = ("NO LIFT — specialization didn't help held-out (forgetting or task ceiling) -> lead with the "
                   "efficiency + descriptor-FM story; real descriptors are the next lever.")
    print(f"  VERDICT: {verdict}")
    print("=" * 64)

    RESULT_JSON.write_text(json.dumps({
        "held_classes": len(held_classes), "chance": chance,
        "base_acc": base_acc, "specialized_acc": spec_acc, "best_epoch_acc": best,
        "lift": lift, "teacher_acc": teach_acc,
        "base_by_crop": base_by, "specialized_by_crop": spec_by,
        "config": {"student": f"{sname}/{spre}", "teacher": f"{tname}/{tpre}",
                   "alpha": ALPHA, "tau": TAU, "lambda_kd": LAMBDA_KD, "epochs": EPOCHS, "lr": LR},
    }, indent=2))
    print(f"saved {RESULT_JSON}")


if __name__ == "__main__":
    main()
