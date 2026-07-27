"""
temp/bakeoff.py — encoder/teacher bake-off for cross-crop zero-shot (EXP 1).

Which image-text encoder is the best zero-shot agricultural diagnoser? Compares our MobileCLIP
family + SigLIP2 (confirmed) + BioCLIP2 + SCOLD (+AgriCLIP best-effort) on the held-out crops with
RICH descriptors. open_clip models load uniformly; SCOLD uses a custom transformers adapter (Swin-T +
RoBERTa) that probes its API and skips cleanly if it can't be driven.

Run on Kaggle (GPU + Internet ON):
    !git clone https://github.com/Abhiram970/plant-disease-edge.git
    %cd plant-disease-edge
    !PDE_DATA_ROOT=/kaggle/working python temp/bakeoff.py
"""
from __future__ import annotations
import importlib
import json
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

# (label, kind, name, pretrained)
ENCODERS = [
    ("MobileCLIP2-S0", "openclip", "MobileCLIP2-S0", "dfndr2b"),
    ("MobileCLIP-S1",  "openclip", "MobileCLIP-S1",  "datacompdr"),
    ("MobileCLIP2-S2", "openclip", "MobileCLIP2-S2", "dfndr2b"),
    ("SigLIP2",        "openclip", "ViT-B-16-SigLIP2", "webli"),
    ("BioCLIP2",       "openclip", "hf-hub:imageomics/bioclip-2", None),
    ("SCOLD",          "scold",    "enalis/scold",   None),
    ("AgriCLIP",       "openclip", "hf-hub:MBZUAI/AgriCLIP", None),   # best-effort; may not exist
]


def ensure_deps():
    need = []
    for mod, pkg in [("open_clip", "open_clip_torch>=2.24"), ("timm", "timm>=1.0.3"),
                     ("transformers", "transformers>=4.40"), ("torchvision", "torchvision"),
                     ("huggingface_hub", "huggingface_hub"), ("pyarrow", "pyarrow"), ("tqdm", "tqdm")]:
        try:
            importlib.import_module(mod)
        except Exception:
            need.append(pkg)
    if need:
        print(f"[deps] installing {need}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)


class OpenClipEnc:
    def __init__(self, name, pretrained, device):
        import open_clip
        if pretrained:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(name)
        self.tok = open_clip.get_tokenizer(name)
        self.model.eval().to(device)
        self.device = device
        self.img_params_m = sum(p.numel() for p in self.model.visual.parameters()) / 1e6

    def encode_image(self, px):
        import torch, torch.nn.functional as F
        with torch.no_grad():
            return F.normalize(self.model.encode_image(px.to(self.device)), dim=-1)

    def encode_text(self, texts):
        import torch, torch.nn.functional as F
        with torch.no_grad():
            return F.normalize(self.model.encode_text(self.tok(texts).to(self.device)), dim=-1)


class ScoldEnc:
    """Best-effort SCOLD adapter (transformers; Swin-T + RoBERTa). Probes common CLIP-style APIs."""
    def __init__(self, repo, device):
        from transformers import AutoModel, AutoTokenizer
        from torchvision import transforms as T
        self.device = device
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval().to(device)
        self.tok = AutoTokenizer.from_pretrained(repo)
        try:
            from transformers import AutoImageProcessor
            self.proc = AutoImageProcessor.from_pretrained(repo)
        except Exception:
            self.proc = None
        self.preprocess = (self._proc if self.proc else
                           T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                                      T.Normalize([0.481, 0.458, 0.408], [0.269, 0.261, 0.276])]))
        self.img_params_m = sum(p.numel() for p in self.model.parameters()) / 1e6  # whole-model (rough)

    def _proc(self, pil):
        return self.proc(pil, return_tensors="pt")["pixel_values"][0]

    def encode_image(self, px):
        import torch, torch.nn.functional as F
        px = px.to(self.device)
        with torch.no_grad():
            for fn in ("get_image_features", "encode_image"):
                if hasattr(self.model, fn):
                    return F.normalize(getattr(self.model, fn)(px), dim=-1)
            out = self.model(pixel_values=px)
            emb = getattr(out, "image_embeds", out)
            return F.normalize(emb, dim=-1)

    def encode_text(self, texts):
        import torch, torch.nn.functional as F
        t = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            for fn in ("get_text_features", "encode_text"):
                if hasattr(self.model, fn):
                    try:
                        emb = getattr(self.model, fn)(input_ids=t["input_ids"], attention_mask=t.get("attention_mask"))
                    except TypeError:
                        emb = getattr(self.model, fn)(t["input_ids"])
                    return F.normalize(emb, dim=-1)
            out = self.model(input_ids=t["input_ids"], attention_mask=t.get("attention_mask"))
            return F.normalize(getattr(out, "text_embeds", out), dim=-1)


def embed_images(enc, rows, batch=64):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    class DS(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            return enc.preprocess(Image.open(rows[i]["path"]).convert("RGB")), rows[i]["label"]

    dl = DataLoader(DS(), batch_size=batch, num_workers=2)
    embs, labels = [], []
    for px, lab in dl:
        embs.append(enc.encode_image(px).cpu())
        labels += list(lab)
    return torch.cat(embs), labels


def build_protos(enc, classes, strategy="rich"):
    import torch, torch.nn.functional as F
    protos = []
    for c in classes:
        texts = [t.format(D.text_for(c, strategy)) for t in C.PROMPT_TEMPLATES]
        protos.append(enc.encode_text(texts).mean(0))
    return F.normalize(torch.stack(protos), dim=-1).cpu()


def acc_of(embs, protos, labels, classes):
    pred = (embs @ protos.T).argmax(1).tolist()
    per = defaultdict(lambda: [0, 0]); ok = tot = 0
    for p, gt in zip(pred, labels):
        hit = classes[p] == gt; ok += hit; tot += 1
        cr = gt.split("|")[0]; per[cr][0] += hit; per[cr][1] += 1
    return ok / tot, {c: a / n for c, (a, n) in per.items()}


def main():
    ensure_deps()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = sage_data.fetch(C.HELDOUT_CROPS, sage_data.full_caps(), min_held_crops=C.MIN_HELD_CROPS)
    assert rows, "no held-out images"
    classes = sorted({r["label"] for r in rows})
    chance = 1.0 / len(classes)
    print(f"[bakeoff] held={len(rows):,} imgs  {len(classes)} classes  chance={chance:.1%}\n")

    results = {}
    for label, kind, name, pre in ENCODERS:
        try:
            enc = OpenClipEnc(name, pre, device) if kind == "openclip" else ScoldEnc(name, device)
        except Exception as e:
            print(f"  {label:14s} LOAD failed  ({type(e).__name__}: {str(e)[:70]})")
            continue
        try:
            embs, labels = embed_images(enc, rows)
            protos = build_protos(enc, classes, "rich")
            acc, by = acc_of(embs, protos, labels, classes)
            results[label] = {"img_params_M": round(enc.img_params_m, 1), "rich_acc": acc, "by_crop": by}
            print(f"  {label:14s} ~{enc.img_params_m:6.1f}M  rich_zeroshot={acc:5.1%}  "
                  + ", ".join(f"{c}={v:.0%}" for c, v in by.items()))
        except Exception as e:
            print(f"  {label:14s} EVAL failed  ({type(e).__name__}: {str(e)[:70]})")
        del enc
        if device == "cuda":
            torch.cuda.empty_cache()

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS_DIR / "bakeoff.json"
    out.write_text(json.dumps({"chance": chance, "n_classes": len(classes),
                               "crops": sorted({c.split('|')[0] for c in classes}), "models": results}, indent=2))
    print(f"\n[bakeoff] saved {out}")
    print("READ: highest rich_zeroshot = best encoder. If SCOLD/BioCLIP2 > SigLIP2 -> domain teacher wins "
          "(lifts baseline + connects us to 2026 work). If SCOLD LOAD failed -> paste its model-card snippet.")


if __name__ == "__main__":
    main()
