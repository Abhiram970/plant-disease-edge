"""
Phase 0 DESCRIPTORS — does descriptor QUALITY move frozen cross-crop zero-shot?
==============================================================================
Every run so far used 14 crude keyword stubs. SAGE says source-grounded symptom
text adds +14-16pp. This tests that lever -- eval-only, NO training -- by comparing
three text-prototype strategies on the FROZEN models:
    bare  = "{disease} on {crop} leaf"            (class name only)
    crude = bare + one generic keyword sentence   (what we used so far)
    rich  = bare + a detailed per-disease symptom description (source-grounded STYLE)

If rich >> crude on the frozen MobileCLIP2-S0 (11M) and SigLIP2, the descriptor-driven
edge headline is real and the paper's novel lever (source-grounded descriptors) is
validated. If rich ~ crude, descriptors are near their ceiling and we lead with efficiency.

NOTE: the rich text here is authored from agronomic knowledge to test RICHNESS; the real
pipeline replaces it with auditable {value, source_url, verbatim_quote} descriptors (Phase A2).

Reuses on-disk held images from phase0_spike.py. Run on Kaggle (GPU, Internet ON).
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("/kaggle/working/spike_data")
if not DATA_DIR.exists():
    DATA_DIR = Path("./phase0_out/spike_data")
HELDOUT_CROPS = ["Coffee", "Orange", "Peach"]
MIN_CLASS_IMAGES = 15
RESULT_JSON = Path("/kaggle/working/phase0_descriptors_result.json")

# --- standalone data fetch (so this works in a FRESH Kaggle session with empty /kaggle/working) ---
# Front-load shard 0 (small) + shard 8 (known to hold Peach); auto-stop once all 3 held crops covered.
SHARD_ORDER = [0, 8, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
MAX_SHARDS = 13
CAP_HELD_PER_CLASS = 120
CROP_ALIASES = {"coffee": "Coffee", "orange": "Orange", "citrus": "Orange",
                "sweet orange": "Orange", "peach": "Peach"}

MODELS = [("MobileCLIP2-S0", "dfndr2b"), ("MobileCLIP-S1", "datacompdr"), ("ViT-B-16-SigLIP2", "webli")]
TEMPLATES = ["a photo of {}", "a close-up leaf photo: {}", "a leaf with {}"]

# crude keyword stubs (what every prior run used)
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

# rich per-disease symptom descriptions (source-grounded STYLE; priority-ordered, specific first)
RICH = [
    ("huanglongbing", "blotchy asymmetric yellow mottling that does not mirror across the midrib, with "
                      "yellowed veins and a thickened leathery leaf; a hallmark of citrus greening"),
    ("greening", "blotchy asymmetric yellow mottling across the leaf, not matching on either side of the "
                 "midrib, with green islands and yellow veins"),
    ("citrus canker", "raised tan-to-brown corky lesions ringed by a yellow halo and a water-soaked margin "
                      "on the leaf"),
    ("leaf curl", "severely thickened, puckered and curled leaves, reddish to purple, later developing a "
                  "whitish powdery bloom and turning yellow"),
    ("brown rot", "rapidly spreading firm brown rot bearing tufts of tan-grey powdery spores on the tissue"),
    ("black spot", "small dark sunken circular spots with pale grey centres and a brittle cracked surface"),
    ("brown eye", "circular tan-to-brown spots with pale grey or white centres surrounded by a yellow halo"),
    ("cercospora", "circular brown spots with grey centres ringed by a bright yellow halo on the leaf"),
    ("leaf miner", "winding translucent serpentine mines and silvery tunnels meandering within the leaf tissue"),
    ("red spider", "fine pale stippling and dull bronzing of the leaf with faint webbing"),
    ("spider mite", "fine pale stippling and bronzing of the leaf surface with delicate webbing"),
    ("shot hole", "small reddish-purple spots whose centres drop out to leave clean round shot holes in the leaf"),
    ("bacterial spot", "small angular dark purple-to-brown spots confined by leaf veins, often dropping out "
                       "to a shot-hole look, with yellowing around them"),
    ("powdery mildew", "white powdery fungal patches dusting the leaf surface, distorting and curling young leaves"),
    ("downy mildew", "pale yellow angular blotches on the upper leaf with grey-purple downy mould beneath"),
    ("greasy spot", "yellow blistered mottling on the upper leaf with brown greasy translucent blisters underneath"),
    ("melanose", "numerous tiny raised dark-brown sandpaper-textured specks on young leaves, sometimes in "
                 "tear-streak patterns"),
    ("anthracnose", "sunken dark lesions with concentric rings and a tan papery centre on the leaf"),
    ("phoma", "dark brown to black necrotic blotches at the leaf margins and tips, often with concentric zoning"),
    ("canker", "raised corky brown lesions with a yellow halo on the leaf and stem"),
    ("scab", "raised wart-like corky scabby pustules with cracked, distorted and wrinkled leaf tissue"),
    ("rust", "yellow-orange powdery pustules on the underside of the leaf with matching pale chlorotic "
             "blotches above, coalescing into large yellow areas"),
    ("curl", "puckered, thickened and distorted curled leaves, often reddened"),
    ("mildew", "a white-to-grey powdery fungal coating spreading over the leaf surface"),
    ("mosaic", "a mottled light-and-dark green mosaic with mild puckering of the leaf"),
    ("blight", "rapidly spreading brown necrotic lesions killing large areas of leaf tissue"),
    ("deficiency", "interveinal yellowing of the leaf with veins staying green, from nutrient deficiency"),
    ("nutrient", "interveinal yellowing while the veins remain green, indicating nutrient deficiency"),
    ("mite", "fine pale stippling and bronzing of the leaf with faint webbing"),
    ("spot", "scattered dark circular leaf spots with concentric rings and yellow margins"),
    ("rot", "spreading soft brown rot of the tissue with fungal growth"),
    ("healthy", "a uniformly green, glossy, healthy leaf with no spots, mottling, lesions or distortion"),
]


def ensure_deps():
    import importlib, subprocess
    need = []
    for mod, pkg in [("open_clip", "open_clip_torch>=2.24"), ("timm", "timm>=1.0.3"),
                     ("huggingface_hub", "huggingface_hub"), ("pyarrow", "pyarrow"), ("tqdm", "tqdm")]:
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


def _safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s).strip())


def _canonical_crop(raw):
    if not raw:
        return None
    k = str(raw).strip().lower()
    c = CROP_ALIASES.get(k) or next((v for a, v in CROP_ALIASES.items() if a in k), None)
    return c if c in HELDOUT_CROPS else None


def _load_done():
    p = DATA_DIR / ".shards_done.json"
    if p.exists():
        try:
            return set(json.loads(p.read_text()))
        except Exception:
            return set()
    return set()


def _save_done(done):
    (DATA_DIR / ".shards_done.json").write_text(json.dumps(sorted(done)))


def ensure_held_data():
    """Fetch held-out (Coffee/Orange/Peach) images from SAGE shards if not already on disk,
    so this script runs standalone in a fresh Kaggle session. Incremental + resumable."""
    import io, hashlib
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from tqdm.auto import tqdm

    def crop_of(lbl):
        return lbl.split("|", 1)[0]

    def covered(rs):
        by = Counter(crop_of(r["label"]) for r in rs)
        return all(by.get(c, 0) >= MIN_CLASS_IMAGES for c in HELDOUT_CROPS)

    rows = load_rows(HELDOUT_CROPS, min_imgs=1)
    if covered(rows):
        print(f"[data] held images already on disk ({len(rows):,}) -> skip download.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    done = _load_done()
    kept = Counter((crop_of(r["label"]), r["label"].split("|", 1)[1]) for r in rows)
    hashes = {Path(r["path"]).stem for r in rows}
    print(f"[data] fetching held crops {HELDOUT_CROPS} from SAGE shards (have {len(rows):,}) ...")
    for si in SHARD_ORDER[:MAX_SHARDS]:
        if covered(rows):
            break
        if si in done:
            continue
        fn = f"default/train/{si:04d}.parquet"
        print(f"    downloading shard {si:04d} ...")
        try:
            path = hf_hub_download(repo_id="tirtho149/SAGE", repo_type="dataset",
                                   filename=fn, revision="refs/convert/parquet")
        except Exception as e:
            print(f"    !! shard {si:04d} failed: {e}")
            continue
        pf = pq.ParquetFile(path)
        try:
            names = set(pf.schema_arrow.names)
            cols = [c for c in ("image", "crop", "disease") if c in names]
        except Exception:
            cols = None
        for batch in tqdm(pf.iter_batches(batch_size=512, columns=cols),
                          total=pf.metadata.num_rows // 512 + 1, desc=f"shard{si:04d}"):
            d = batch.to_pydict()
            imgs = d.get("image", []); crops = d.get("crop", [])
            diss = d.get("disease", [None] * len(crops))
            for img_obj, craw, draw in zip(imgs, crops, diss):
                crop = _canonical_crop(craw)
                if crop is None:
                    continue
                disease = str(draw if draw is not None else "Unknown")
                key = (crop, disease)
                if kept[key] >= CAP_HELD_PER_CLASS:
                    continue
                try:
                    raw = img_obj["bytes"] if isinstance(img_obj, dict) else img_obj
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=92)
                    jpg = buf.getvalue()
                except Exception:
                    continue
                h16 = hashlib.sha1(jpg).hexdigest()[:16]
                if h16 in hashes:
                    continue
                hashes.add(h16)
                cls = DATA_DIR / f"{_safe(crop)}___{_safe(disease)}"
                cls.mkdir(parents=True, exist_ok=True)
                fp = cls / f"{h16}.jpg"
                fp.write_bytes(jpg)
                kept[key] += 1
                rows.append({"path": str(fp), "label": f"{crop}|{disease}"})
        try:
            Path(path).unlink()
        except Exception:
            pass
        done.add(si); _save_done(done)
        by = Counter(crop_of(r["label"]) for r in rows)
        print(f"    after shard {si:04d}: " + ", ".join(f"{c}={by.get(c,0)}" for c in HELDOUT_CROPS))
    if not covered(rows):
        by = Counter(crop_of(r["label"]) for r in rows)
        print(f"    WARNING: held crops under {MIN_CLASS_IMAGES} imgs: "
              + ", ".join(f"{c}={by.get(c,0)}" for c in HELDOUT_CROPS))


def text_for(lbl, strategy, coverage=None):
    crop, dis = lbl.split("|")
    base = f"{dis} on {crop} leaf".replace("_", " ")
    k = dis.lower()
    if strategy == "bare":
        return base
    if strategy == "crude":
        hint = next((v for kw, v in CRUDE.items() if kw in k), "")
        return f"{base}: {hint}" if hint else base
    # rich
    for kw, desc in RICH:
        if kw in k:
            if coverage is not None:
                coverage[lbl] = kw
            return f"{base}. {desc}"
    if coverage is not None:
        coverage[lbl] = "(NO MATCH)"
    return base


def main():
    ensure_deps()
    ensure_held_data()        # fetch Coffee/Orange/Peach from SAGE if not already on disk
    import torch
    import torch.nn.functional as F
    import open_clip
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    held = load_rows(HELDOUT_CROPS)
    assert held, "no held images on disk"
    classes = sorted({r["label"] for r in held})
    chance = 1.0 / len(classes)
    print(f"[*] held={len(held):,} imgs  {len(classes)} classes  chance={chance:.1%}")

    # show rich-descriptor coverage so we can spot misses
    cov = {}
    for c in classes:
        text_for(c, "rich", cov)
    print("[*] rich-descriptor coverage:")
    for c in classes:
        print(f"      {c:38s} -> {cov[c]}")
    print()

    def zshot(emb, protos, labels):
        pred = (emb.to(dev) @ protos.T).argmax(1).cpu().tolist()
        per = defaultdict(lambda: [0, 0]); ok = tot = 0
        for p, gt in zip(pred, labels):
            hit = classes[p] == gt; ok += hit; tot += 1
            cr = gt.split("|")[0]; per[cr][0] += hit; per[cr][1] += 1
        return ok / tot, {c: a / n for c, (a, n) in per.items()}

    results = {}
    for name, pre in MODELS:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre)
            tok = open_clip.get_tokenizer(name)
        except Exception as e:
            print(f"  {name} unavailable -> skip ({type(e).__name__})")
            continue
        model.eval().to(dev)

        class DS(Dataset):
            def __len__(self): return len(held)
            def __getitem__(self, i):
                return preprocess(Image.open(held[i]["path"]).convert("RGB")), held[i]["label"]

        dl = DataLoader(DS(), batch_size=128, num_workers=2)
        embs, labels = [], []
        with torch.no_grad():
            for imgs, lab in dl:
                embs.append(F.normalize(model.encode_image(imgs.to(dev)), dim=-1).cpu())
                labels += list(lab)
        embs = torch.cat(embs)

        row = {}
        for strat in ("bare", "crude", "rich"):
            with torch.no_grad():
                protos = []
                for c in classes:
                    toks = tok([t.format(text_for(c, strat)) for t in TEMPLATES]).to(dev)
                    e = F.normalize(model.encode_text(toks), dim=-1).mean(0)
                    protos.append(F.normalize(e, dim=-1))
                protos = torch.stack(protos).to(dev)
            acc, by = zshot(embs, protos, labels)
            row[strat] = {"acc": acc, "by_crop": by}
        results[f"{name}/{pre}"] = row
        print(f"  {name:18s}  bare={row['bare']['acc']:5.1%}   crude={row['crude']['acc']:5.1%}   "
              f"rich={row['rich']['acc']:5.1%}   (chance {chance:.1%})")

    RESULT_JSON.write_text(json.dumps(
        {"chance": chance, "n_classes": len(classes), "coverage": cov, "models": results}, indent=2))
    print(f"\n[probe] saved {RESULT_JSON}")
    print("READ: rich >> crude  => source-grounded descriptors are the lever; descriptor-driven edge headline is real.")
    print("      rich ~ crude   => descriptors near ceiling; lead with the efficiency + flat-curve story.")


if __name__ == "__main__":
    main()
