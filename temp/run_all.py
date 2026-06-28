"""
temp/run_all.py — SELF-CONTAINED Kaggle script (no repo clone / no git auth needed).
Runs EXP1 (encoder bake-off) + EXP2 (train seen / keep unseen) in one go.

UPLOAD THIS ONE FILE and run on Kaggle (GPU T4 + Internet ON):
    %run run_all.py
    # or:  !python run_all.py            (EXP2 tier:  !python run_all.py --tier lw11)

It installs deps, fetches SAGE held + train crops itself (parquet shards, incremental,
mining all wanted crops), then:
  EXP1  — bake-off: MobileCLIP family + SigLIP2 + BioCLIP2 + SCOLD(+AgriCLIP) zero-shot on held crops
  EXP2  — linear probe on SEEN crops (frozen backbone) vs zero-shot on seen + UNSEEN zero-shot preserved
Saves /kaggle/working/run_all_bakeoff.json and run_all_train_seen.json.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import io
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ----------------------------- CONFIG ----------------------------- #
ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./exp_out")
DATA_DIR = ROOT / "exp_data"          # <Crop>___<Disease>/<hash>.jpg
DONE_PATH = DATA_DIR / ".shards_done.json"

SAGE_REPO = "tirtho149/SAGE"
SHARD_REVISION = "refs/convert/parquet"
SHARD_ORDER = [0, 8, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
MAX_SHARDS = 13

TRAIN_CROPS = ["Tomato", "Soybean", "Apple", "Corn", "Grape", "Potato", "Rice"]
HELDOUT_CROPS = ["Coffee", "Orange", "Peach"]
CROP_ALIASES = {
    "tomato": "Tomato", "soybean": "Soybean", "soya": "Soybean", "soya bean": "Soybean",
    "apple": "Apple", "corn": "Corn", "maize": "Corn", "corn (maize)": "Corn",
    "grape": "Grape", "grapevine": "Grape", "potato": "Potato", "rice": "Rice",
    "coffee": "Coffee", "orange": "Orange", "citrus": "Orange", "sweet orange": "Orange",
    "peach": "Peach",
}
CAP_TRAIN, CAP_HELD = 250, 120
MIN_CLASS_IMAGES, MIN_HELD_CROPS, MIN_TRAIN_CROPS = 15, 2, 4

MODEL_TIERS = {"lw11": ("MobileCLIP2-S0", "dfndr2b"),
               "lw21": ("MobileCLIP-S1", "datacompdr"),
               "lw35": ("MobileCLIP2-S2", "dfndr2b")}
ENCODERS = [   # (label, kind, name, pretrained)
    ("MobileCLIP2-S0", "openclip", "MobileCLIP2-S0", "dfndr2b"),
    ("MobileCLIP-S1",  "openclip", "MobileCLIP-S1",  "datacompdr"),
    ("MobileCLIP2-S2", "openclip", "MobileCLIP2-S2", "dfndr2b"),
    ("SigLIP2",        "openclip", "ViT-B-16-SigLIP2", "webli"),
    ("BioCLIP2",       "openclip", "hf-hub:imageomics/bioclip-2", None),
    ("SCOLD",          "scold",    "enalis/scold",   None),
    ("AgriCLIP",       "openclip", "hf-hub:MBZUAI/AgriCLIP", None),
]
PROMPT_TEMPLATES = ["a photo of {}", "a close-up leaf photo: {}", "a leaf with {}"]


def full_caps():
    return {**{c: CAP_TRAIN for c in TRAIN_CROPS}, **{c: CAP_HELD for c in HELDOUT_CROPS}}


def canonical_crop(raw):
    if not raw:
        return None
    k = str(raw).strip().lower()
    c = CROP_ALIASES.get(k) or next((v for a, v in CROP_ALIASES.items() if a in k), None)
    return c


def safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s).strip())


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


# ----------------------------- DATA ----------------------------- #
def load_rows(crops, min_imgs=1):
    rows = []
    if not DATA_DIR.exists():
        return rows
    want = set(crops)
    for d in DATA_DIR.iterdir():
        if not d.is_dir() or "___" not in d.name:
            continue
        crop, disease = d.name.split("___", 1)
        if crop not in want:
            continue
        for jpg in d.glob("*.jpg"):
            rows.append({"path": str(jpg), "crop": crop, "disease": disease, "label": f"{crop}|{disease}"})
    cc = Counter(r["label"] for r in rows)
    return [r for r in rows if cc[r["label"]] >= min_imgs]


def _load_done():
    if DONE_PATH.exists():
        try:
            return set(json.loads(DONE_PATH.read_text()))
        except Exception:
            return set()
    return set()


def fetch(stop_crops, min_stop_crops):
    """Download shards until >= min_stop_crops of stop_crops are covered; mine ALL wanted crops."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from tqdm.auto import tqdm

    caps = full_caps()

    def covered(rows):
        by = Counter(r["crop"] for r in rows)
        return sum(1 for c in stop_crops if by.get(c, 0) >= MIN_CLASS_IMAGES) >= min_stop_crops

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(list(caps), min_imgs=1)
    if covered(rows):
        print(f"[data] enough on disk for {stop_crops} -> skip download.")
        return load_rows(stop_crops, MIN_CLASS_IMAGES)

    done = _load_done()
    kept = Counter((r["crop"], r["disease"]) for r in rows)
    hashes = {Path(r["path"]).stem for r in rows}
    print(f"[data] fetching to cover >= {min_stop_crops} of {stop_crops} (have {len(rows):,}) ...")
    for si in SHARD_ORDER[:MAX_SHARDS]:
        if covered(rows):
            break
        if si in done:
            continue
        fn = f"default/train/{si:04d}.parquet"
        print(f"    shard {si:04d} ...")
        try:
            path = hf_hub_download(repo_id=SAGE_REPO, repo_type="dataset", filename=fn, revision=SHARD_REVISION)
        except Exception as e:
            print(f"    !! shard {si:04d} failed: {e}")
            continue
        pf = pq.ParquetFile(path)
        try:
            cols = [c for c in ("image", "crop", "disease") if c in set(pf.schema_arrow.names)]
        except Exception:
            cols = None
        for batch in tqdm(pf.iter_batches(batch_size=512, columns=cols),
                          total=pf.metadata.num_rows // 512 + 1, desc=f"shard{si:04d}"):
            d = batch.to_pydict()
            imgs = d.get("image", []); crps = d.get("crop", []); diss = d.get("disease", [None] * len(crps))
            for img_obj, craw, draw in zip(imgs, crps, diss):
                crop = canonical_crop(craw)
                if crop is None or crop not in caps:
                    continue
                disease = str(draw if draw is not None else "Unknown")
                key = (crop, disease)
                if kept[key] >= caps[crop]:
                    continue
                try:
                    raw = img_obj["bytes"] if isinstance(img_obj, dict) else img_obj
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=92); jpg = buf.getvalue()
                except Exception:
                    continue
                h16 = hashlib.sha1(jpg).hexdigest()[:16]
                if h16 in hashes:
                    continue
                hashes.add(h16)
                cls = DATA_DIR / f"{safe(crop)}___{safe(disease)}"
                cls.mkdir(parents=True, exist_ok=True)
                (cls / f"{h16}.jpg").write_bytes(jpg)
                kept[key] += 1
                rows.append({"path": str(cls / f"{h16}.jpg"), "crop": crop, "disease": disease,
                             "label": f"{crop}|{disease}"})
        try:
            Path(path).unlink()
        except Exception:
            pass
        done.add(si); DONE_PATH.write_text(json.dumps(sorted(done)))
        by = Counter(r["crop"] for r in rows)
        print("    counts: " + ", ".join(f"{c}={by.get(c, 0)}" for c in (TRAIN_CROPS + HELDOUT_CROPS) if by.get(c, 0)))
    return load_rows(stop_crops, MIN_CLASS_IMAGES)


# ----------------------------- DESCRIPTORS ----------------------------- #
CRUDE = {
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
RICH = [
    ("huanglongbing", "blotchy asymmetric yellow mottling that does not mirror across the midrib, with yellowed veins and a thickened leathery leaf; a hallmark of citrus greening"),
    ("greening", "blotchy asymmetric yellow mottling across the leaf, not matching on either side of the midrib, with green islands and yellow veins"),
    ("citrus canker", "raised tan-to-brown corky lesions ringed by a yellow halo and a water-soaked margin"),
    ("leaf curl", "severely thickened, puckered and curled leaves, reddish to purple, later with a whitish bloom"),
    ("brown rot", "rapidly spreading firm brown rot bearing tufts of tan-grey powdery spores"),
    ("black spot", "small dark sunken circular spots with pale grey centres and a brittle cracked surface"),
    ("brown eye", "circular tan-to-brown spots with pale grey or white centres surrounded by a yellow halo"),
    ("cercospora", "circular brown spots with grey centres ringed by a bright yellow halo on the leaf"),
    ("leaf miner", "winding translucent serpentine mines and silvery tunnels within the leaf tissue"),
    ("red spider", "fine pale stippling and dull bronzing of the leaf with faint webbing"),
    ("spider mite", "fine pale stippling and bronzing of the leaf surface with delicate webbing"),
    ("shot hole", "small reddish-purple spots whose centres drop out to leave clean round shot holes"),
    ("bacterial spot", "small angular dark purple-to-brown spots confined by veins, often dropping to shot-holes"),
    ("powdery mildew", "white powdery fungal patches dusting the leaf surface, distorting young leaves"),
    ("downy mildew", "pale yellow angular blotches on the upper leaf with grey-purple downy mould beneath"),
    ("greasy spot", "yellow blistered mottling above with brown greasy translucent blisters underneath"),
    ("melanose", "numerous tiny raised dark-brown sandpaper-textured specks, sometimes in tear-streaks"),
    ("anthracnose", "sunken dark lesions with concentric rings and a tan papery centre"),
    ("phoma", "dark brown to black necrotic blotches at leaf margins and tips, with concentric zoning"),
    ("canker", "raised corky brown lesions with a yellow halo on the leaf and stem"),
    ("scab", "raised wart-like corky scabby pustules with cracked, distorted, wrinkled tissue"),
    ("rust", "yellow-orange powdery pustules under the leaf with matching pale chlorotic blotches above"),
    ("curl", "puckered, thickened and distorted curled leaves, often reddened"),
    ("mildew", "a white-to-grey powdery fungal coating spreading over the leaf surface"),
    ("mosaic", "a mottled light-and-dark green mosaic with mild puckering"),
    ("blight", "rapidly spreading brown necrotic lesions killing large areas of leaf tissue"),
    ("deficiency", "interveinal yellowing with the veins staying green, from nutrient deficiency"),
    ("nutrient", "interveinal yellowing while veins remain green, indicating nutrient deficiency"),
    ("mite", "fine pale stippling and bronzing of the leaf with faint webbing"),
    ("spot", "scattered dark circular leaf spots with concentric rings and yellow margins"),
    ("rot", "spreading soft brown rot of the tissue with fungal growth"),
    ("healthy", "a uniformly green, glossy, healthy leaf with no spots, mottling, lesions or distortion"),
]


def text_for(label, strategy="rich", coverage=None):
    crop, dis = label.split("|", 1)
    base = f"{dis} on {crop} leaf".replace("_", " ")
    k = dis.lower()
    if strategy == "bare":
        return base
    if strategy == "crude":
        hint = next((v for kw, v in CRUDE.items() if kw in k), "")
        return f"{base}: {hint}" if hint else base
    for kw, desc in RICH:
        if kw in k:
            if coverage is not None:
                coverage[label] = kw
            return f"{base}. {desc}"
    if coverage is not None:
        coverage[label] = "(NO MATCH)"
    return base


# ----------------------------- ENCODERS ----------------------------- #
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
        import torch.nn.functional as F
        import torch
        with torch.no_grad():
            return F.normalize(self.model.encode_image(px.to(self.device)), dim=-1)

    def encode_text(self, texts):
        import torch.nn.functional as F
        import torch
        with torch.no_grad():
            return F.normalize(self.model.encode_text(self.tok(texts).to(self.device)), dim=-1)


class ScoldEnc:
    def __init__(self, repo, device):
        import torch
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
        self.img_params_m = sum(p.numel() for p in self.model.parameters()) / 1e6

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
            return F.normalize(getattr(out, "image_embeds", out), dim=-1)

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
    protos = [enc.encode_text([t.format(text_for(c, strategy)) for t in PROMPT_TEMPLATES]).mean(0) for c in classes]
    return F.normalize(torch.stack(protos), dim=-1).cpu()


def acc_of(embs, protos, labels, classes):
    pred = (embs @ protos.T).argmax(1).tolist()
    per = defaultdict(lambda: [0, 0]); ok = tot = 0
    for p, gt in zip(pred, labels):
        hit = classes[p] == gt; ok += hit; tot += 1
        cr = gt.split("|")[0]; per[cr][0] += hit; per[cr][1] += 1
    return ok / tot, {c: a / n for c, (a, n) in per.items()}


# ----------------------------- EXPERIMENTS ----------------------------- #
def exp1_bakeoff(held, device):
    classes = sorted({r["label"] for r in held})
    chance = 1.0 / len(classes)
    print(f"\n===== EXP1: encoder bake-off  (held {len(held):,} imgs, {len(classes)} classes, chance {chance:.1%}) =====")
    import torch
    results = {}
    for label, kind, name, pre in ENCODERS:
        try:
            enc = OpenClipEnc(name, pre, device) if kind == "openclip" else ScoldEnc(name, device)
        except Exception as e:
            print(f"  {label:14s} LOAD failed ({type(e).__name__}: {str(e)[:60]})")
            continue
        try:
            embs, labels = embed_images(enc, held)
            acc, by = acc_of(embs, build_protos(enc, classes, "rich"), labels, classes)
            results[label] = {"img_params_M": round(enc.img_params_m, 1), "rich_acc": acc, "by_crop": by}
            print(f"  {label:14s} ~{enc.img_params_m:6.1f}M  rich_zeroshot={acc:5.1%}  "
                  + ", ".join(f"{c}={v:.0%}" for c, v in by.items()))
        except Exception as e:
            print(f"  {label:14s} EVAL failed ({type(e).__name__}: {str(e)[:60]})")
        del enc
        if device == "cuda":
            torch.cuda.empty_cache()
    (ROOT / "run_all_bakeoff.json").write_text(json.dumps(
        {"chance": chance, "n_classes": len(classes), "models": results}, indent=2))
    print(f"  saved {ROOT/'run_all_bakeoff.json'}")
    return results


def exp2_train_seen(tier, seen, unseen, device, epochs=40):
    import torch, torch.nn as nn, torch.nn.functional as F
    name, pre = MODEL_TIERS[tier]
    print(f"\n===== EXP2: train seen / keep unseen  (tier {tier} = {name}) =====")
    enc = OpenClipEnc(name, pre, device)
    seen_emb, seen_lab = embed_images(enc, seen)
    uns_emb, uns_lab = embed_images(enc, unseen)

    seen_classes = sorted({r["label"] for r in seen})
    sidx = {c: i for i, c in enumerate(seen_classes)}
    random.seed(42)
    by_cls = defaultdict(list)
    for i, l in enumerate(seen_lab):
        by_cls[l].append(i)
    tr, te = [], []
    for l, idxs in by_cls.items():
        random.shuffle(idxs); k = max(1, int(0.2 * len(idxs))); te += idxs[:k]; tr += idxs[k:]
    Xtr = seen_emb[tr].to(device); ytr = torch.tensor([sidx[seen_lab[i]] for i in tr], device=device)
    Xte = seen_emb[te]; yte = [seen_lab[i] for i in te]

    clf = nn.Linear(seen_emb.shape[1], len(seen_classes)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"  seen {len(seen_classes)} classes, train {len(tr):,}/test {len(te):,}; probe x {epochs} epochs ...")
    for ep in range(epochs):
        clf.train(); perm = torch.randperm(len(tr), device=device)
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            loss = F.cross_entropy(clf(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(Xte.to(device)).argmax(1).cpu().tolist()
    seen_probe = sum(seen_classes[p] == gt for p, gt in zip(pred, yte)) / len(yte)
    seen_zs, _ = acc_of(seen_emb[te], build_protos(enc, seen_classes, "rich"), yte, seen_classes)

    uns_classes = sorted({r["label"] for r in unseen})
    uns_zs, uns_by = acc_of(uns_emb, build_protos(enc, uns_classes, "rich"), uns_lab, uns_classes)

    torch.save({"tier": tier, "classes": seen_classes, "state_dict": clf.state_dict()},
               ROOT / f"run_all_probe_{tier}.pt")
    print("  " + "-" * 56)
    print(f"  SEEN  trained head : {seen_probe:.1%}   vs zero-shot {seen_zs:.1%}   (training wins on seen)")
    print(f"  UNSEEN zero-shot   : {uns_zs:.1%}   (frozen backbone -> preserved)   "
          + ", ".join(f"{c}={v:.0%}" for c, v in uns_by.items()))
    (ROOT / f"run_all_train_seen_{tier}.json").write_text(json.dumps(
        {"tier": tier, "seen_classes": len(seen_classes), "unseen_classes": len(uns_classes),
         "seen_probe_acc": seen_probe, "seen_zeroshot_acc": seen_zs,
         "unseen_zeroshot_acc": uns_zs, "unseen_by_crop": uns_by}, indent=2))
    print(f"  saved {ROOT/('run_all_train_seen_'+tier+'.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="lw11", choices=list(MODEL_TIERS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--only", choices=["exp1", "exp2"], default=None)
    args = ap.parse_args()

    ensure_deps()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] device={device}")

    held = fetch(HELDOUT_CROPS, MIN_HELD_CROPS)
    if args.only != "exp2":
        exp1_bakeoff(held, device)
    if args.only != "exp1":
        seen = fetch(TRAIN_CROPS, MIN_TRAIN_CROPS)
        exp2_train_seen(args.tier, seen, held, device, args.epochs)
    print("\n[done] result JSONs in /kaggle/working/ (run_all_*.json)")


if __name__ == "__main__":
    main()
