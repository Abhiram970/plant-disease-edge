"""
Phase 0 PROBE — can a small PRETRAINED image-text model do cross-crop disease
zero-shot OUT OF THE BOX?  (eval-only, no training, ~10 min)
=============================================================================
Four distill runs showed an ImageNet backbone (edgenext, ~5M) distilled FROM
SCRATCH on ~10k images does NOT learn cross-crop zero-shot: the student stayed
~chance (11%) even with a SigLIP2 teacher + text anchoring, and a stronger
teacher did NOT help (retention fell). Conclusion: a tiny model cannot absorb
image-text alignment from 10k images -- it must INHERIT it from pretraining.

This probes the literature-backed alternative: models that are ALREADY image-text
aligned at edge size (MobileCLIP / TinyCLIP). We measure their zero-shot accuracy
on the held-out crops DIRECTLY -- no training.

DECISION:
  - If a ~8-35M model clears chance (approaches the SigLIP2 ceiling) -> the headline
    is FEASIBLE at edge scale; the student becomes a pretrained small CLIP that we
    later specialize on descriptors. Pivot the student architecture, keep the paper.
  - If even these are ~chance -> the TASK is the limiter -> pivot the headline
    (few-shot cross-crop, or the teacher-level + efficiency/retention story).

Reuses the held-out images already on disk from phase0_spike.py.
Run on Kaggle (GPU, Internet ON):  %run phase0_probe.py
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("/kaggle/working/spike_data")
if not DATA_DIR.exists():
    DATA_DIR = Path("./phase0_out/spike_data")
HELDOUT_CROPS = ["Coffee", "Orange", "Peach"]
RESULT_JSON = Path("/kaggle/working/phase0_probe_result.json")

# Candidate small PRETRAINED image-text models (image-encoder size in comment).
# Plus the SigLIP2 reference ceiling we already measured (~25.6%).
CANDIDATES = [
    ("TinyCLIP-ViT-8M-16-Text-3M", "YFCC15M"),     # ~8M img encoder
    ("MobileCLIP-S0", "datacompdr"),               # ~11M
    ("MobileCLIP-S1", "datacompdr"),               # ~21M
    ("TinyCLIP-ViT-39M-16-Text-19M", "YFCC15M"),   # ~39M
    ("MobileCLIP-S2", "datacompdr"),               # ~35M
    ("ViT-B-16-SigLIP2", "webli"),                 # reference ceiling (the spike teacher)
]

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


def load_held():
    rows = []
    if not DATA_DIR.exists():
        sys.exit(f"no data dir {DATA_DIR}; run phase0_spike.py first to populate held-out images.")
    for d in DATA_DIR.iterdir():
        if not d.is_dir() or "___" not in d.name:
            continue
        crop, disease = d.name.split("___", 1)
        if crop not in HELDOUT_CROPS:
            continue
        for jpg in d.glob("*.jpg"):
            rows.append({"path": str(jpg), "label": f"{crop}|{disease}"})
    return rows


def main():
    ensure_deps()
    import torch
    import torch.nn.functional as Fn
    import open_clip
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    held = load_held()
    assert held, f"no held-out images under {DATA_DIR}"
    classes = sorted({r["label"] for r in held})
    chance = 1.0 / len(classes)
    crops = sorted({c.split("|")[0] for c in classes})
    print(f"[probe] held={len(held):,} imgs  {len(classes)} classes  crops={crops}  chance={chance:.1%}\n")

    results = {}
    for name, pre in CANDIDATES:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre)
            tok = open_clip.get_tokenizer(name)
        except Exception as e:
            print(f"  {name:30s} unavailable ({type(e).__name__}) -> skip")
            continue
        model.eval().to(dev)
        nimg = sum(p.numel() for p in model.visual.parameters()) / 1e6

        with torch.no_grad():
            protos = []
            for lbl in classes:
                toks = tok([t.format(descriptor_text(lbl)) for t in TEMPLATES]).to(dev)
                emb = Fn.normalize(model.encode_text(toks), dim=-1).mean(0)
                protos.append(Fn.normalize(emb, dim=-1))
            protos = torch.stack(protos).to(dev)

        class DS(Dataset):
            def __len__(self): return len(held)
            def __getitem__(self, i):
                r = held[i]
                return preprocess(Image.open(r["path"]).convert("RGB")), r["label"]

        dl = DataLoader(DS(), batch_size=64, num_workers=2)
        correct = total = 0
        per = defaultdict(lambda: [0, 0])
        with torch.no_grad():
            for imgs, labels in dl:
                emb = Fn.normalize(model.encode_image(imgs.to(dev)), dim=-1)
                pred = (emb @ protos.T).argmax(1).cpu().tolist()
                for p, gt in zip(pred, labels):
                    ok = classes[p] == gt
                    correct += ok; total += 1
                    cr = gt.split("|")[0]
                    per[cr][0] += ok; per[cr][1] += 1
        acc = correct / total
        by_crop = {c: a / n for c, (a, n) in per.items()}
        results[f"{name}/{pre}"] = {"img_params_M": round(nimg, 2), "zeroshot_acc": acc,
                                    "by_crop": by_crop}
        print(f"  {name:30s} img={nimg:5.1f}M  zero-shot={acc:5.1%}  (chance {chance:.1%})   "
              + ", ".join(f"{c}={v:.0%}" for c, v in by_crop.items()))

    RESULT_JSON.write_text(json.dumps(
        {"chance": chance, "n_classes": len(classes), "n_held_imgs": len(held),
         "crops": crops, "models": results}, indent=2))
    print(f"\n[probe] saved {RESULT_JSON}")
    print("READ: any model clearly above chance (toward the SigLIP2 reference) => edge zero-shot is "
          "feasible with a PRETRAINED small CLIP student. All ~chance => task is the limiter -> pivot headline.")


if __name__ == "__main__":
    main()
