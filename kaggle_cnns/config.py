"""
Shared configuration for Plant-Disease-Edge (Phase A and beyond).

Single source of truth for crop lists, paths, and dataset coordinates.
Import this everywhere instead of hardcoding. See PROJECT_GUIDE.md sections 2-3.
"""
from __future__ import annotations
import os
from pathlib import Path

try:                                   # make .env authoritative for PDE_* / LAVA_* across all scripts
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Dataset coordinates (SAGE) ---------------------------------------------
SAGE_HF_REPO = "tirtho149/SAGE"          # HuggingFace datasets repo (images, MIT, ~114GB parquet)
SAGE_GH_REPO = "https://github.com/tirtho149/SAGE"  # symptom registry / KB lives here

# --- Crop split — nested scale study (Experiments A ⊂ B ⊂ C) ----------------
# SEEN = crops we train the seen-head on; HELD = crops held out entirely for zero-shot.
# Pools are ordered so a smaller experiment's crops are a PREFIX of the larger one's -> a clean
# controlled "accuracy vs #classes" scaling study, with the anchor crops fixed in their roles.
SEEN_POOL = ["Corn", "Soybean", "Tomato", "Apple",          # A (4) — richest/expert-curated first
             "Grape", "Potato", "Rice", "Sugarcane",        # +B (-> 8)
             "Rose", "Strawberry"]                          # +C (-> 10)
HELD_POOL = ["Coffee", "Orange", "Peach",                   # A (3) — distinctive/economic diseases
             "Cotton", "Wheat", "Bean",                     # +B (-> 6)
             "Banana", "Cucumber"]                          # +C (-> 8) — 8 botanical families, anti-cherry-pick

EXPERIMENTS = {
    "A": {"seen": SEEN_POOL[:4],  "held": HELD_POOL[:3]},   # anchor (already run)
    "B": {"seen": SEEN_POOL[:8],  "held": HELD_POOL[:6]},
    "C": {"seen": SEEN_POOL[:10], "held": HELD_POOL[:8]},
}

# Default = the full pools (Experiment C) so a single fetch grabs every crop; eval scripts
# subset per experiment via EXPERIMENTS and a --exp {A,B,C} flag.
TRAIN_CROPS = SEEN_POOL
HELDOUT_CROPS = HELD_POOL
WANT_CROPS = TRAIN_CROPS + HELDOUT_CROPS   # 18 crops total

# SAGE crop strings vary in spelling/case (e.g. "Corn (Maize)", "Mango Leaf"). Match
# case-insensitively (canonical_crop also does a loose contains-match) -> canonical name.
CROP_ALIASES = {
    "apple": "Apple",
    "corn": "Corn", "maize": "Corn", "corn (maize)": "Corn",
    "potato": "Potato",
    "soybean": "Soybean", "soya": "Soybean", "soya bean": "Soybean",
    "tomato": "Tomato",
    "grape": "Grape", "grapevine": "Grape",
    "rice": "Rice", "paddy": "Rice",
    "rose": "Rose",
    "sugarcane": "Sugarcane", "sugar cane": "Sugarcane",
    "strawberry": "Strawberry",
    "coffee": "Coffee",
    "orange": "Orange", "citrus": "Orange", "sweet orange": "Orange",
    "peach": "Peach",
    "cotton": "Cotton",
    "wheat": "Wheat",
    "bean": "Bean", "common bean": "Bean", "beans": "Bean",
    "banana": "Banana",
    "cucumber": "Cucumber",
}

# --- Data budgets (LOCKED) --------------------------------------------------
PER_CLASS_CAP = 1500       # max images kept per (crop, disease) class
MIN_CLASS_IMAGES = 50      # classes below this are dropped -> OOD/abstain set
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train / val / test (within trained crops)
RANDOM_SEED = 42

# --- Paths ------------------------------------------------------------------
# Override the data root with env var PDE_DATA_ROOT (e.g. on Kaggle: /kaggle/working).
# Default: a ./data folder next to the repo (git-ignored).
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("PDE_DATA_ROOT", REPO_ROOT / "data"))

# Image root. Override with PDE_DATASET_DIR to reuse an existing folder-of-classes build
# (e.g. the run_all.py output at C:\kaggle\working\exp_data) WITHOUT moving 12k files.
DATASET_DIR = Path(os.environ.get("PDE_DATASET_DIR", DATA_ROOT / "dataset_cleaned"))  # <Crop>___<Disease>/<f>.jpg
MANIFEST_CSV = DATA_ROOT / "manifest.csv"     # path, crop, disease, filename, split_role
SPLITS_DIR = DATA_ROOT / "splits"             # train/val/test/heldout/ood .csv
DESCRIPTORS_DIR = REPO_ROOT / "descriptors"   # <crop>.json  (committed to git)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
IMG_SIZE = 224  # student/teacher input resolution

# --- Model family (VALIDATED via Phase-0, 2026-06) --------------------------
# METHOD: students are FROZEN pretrained image-text models (open_clip). Training a tiny model
# to learn the alignment -- from scratch OR by specializing on seen crops -- was shown NOT to
# improve unseen-crop zero-shot (4 runs, catastrophic forgetting). We use frozen backbones +
# descriptor text prototypes. Only the image encoder deploys; text encoder runs offline.
# Deployable LIGHTWEIGHT family (off-the-shelf, FROZEN, probe-validated). Only the image encoder
# ships to the device; the text encoder runs offline to precompute prototypes.
MODEL_TIERS = {
    "lw11": ("MobileCLIP2-S0", "dfndr2b"),      # ~11.4M  lightweight  -> small NPU / phone
    "lw21": ("MobileCLIP-S1",  "datacompdr"),   # ~21.5M  lightweight  -> laptop CPU
    "lw35": ("MobileCLIP2-S2", "dfndr2b"),      # ~35.8M  lightweight  -> laptop
}
HEAVYWEIGHT = ("MobileCLIP-B", "datacompdr")    # ~86.3M  heavyweight (best MobileCLIP-family zero-shot)
REFERENCE   = ("ViT-B-16-SigLIP2", "webli")     # ~92.9M  cloud ceiling / distillation teacher (~25.6%)

# THE 4 DEPLOYABLE MODELS (what we ship + report). SigLIP2 is the reference ceiling, not one of the 4.
# Every downstream script (metrics, benchmark, loco) iterates over these keys; pass --models to subset.
DEPLOY_MODELS = {
    "s0": ("MobileCLIP2-S0", "dfndr2b"),      # ~11.4M  -> small NPU / phone
    "s1": ("MobileCLIP-S1",  "datacompdr"),   # ~21.5M  -> laptop CPU
    "s2": ("MobileCLIP2-S2", "dfndr2b"),      # ~35.8M  -> laptop
    "b":  ("MobileCLIP-B",   "datacompdr"),   # ~86.3M  -> workstation (heavyweight)
}
REFERENCE_MODELS = {"siglip2": REFERENCE}     # ceiling / teacher, evaluated but not deployed


def resolve_models(keys, include_reference=False):
    """Map --models keys -> [(open_clip_name, pretrained)]. Unknown keys are ignored with a note."""
    table = {**DEPLOY_MODELS, **REFERENCE_MODELS}
    out = []
    for k in keys:
        if k in table:
            out.append(table[k])
        else:
            print(f"[config] unknown model key '{k}' (known: {list(table)})")
    if include_reference and REFERENCE not in out:
        out.append(REFERENCE)
    return out

# Frozen teacher VLMs for the bake-off (verify availability on Kaggle open_clip).
TEACHERS = [
    REFERENCE,
    # ("hf-hub:imageomics/bioclip-2", None),  # BioCLIP 2  -- verify load
    # ("hf-hub:enalis/scold", None),          # SCOLD leaf-disease VLM -- verify load
]

# STRETCH tiers = NOT off-the-shelf. There is no aligned model below ~11M, and a gap at ~50M
# (36M -> 86M). Closing the sub-10M gap is framed as a PoC + contribution: TinyCLIP-style
# weight-INHERITED distillation of a lightweight tier down to ~5M. WEAKNESS (honest): a tiny model
# cannot learn cross-modal alignment from limited data (our Phase-0 from-scratch/specialize runs
# confirm); inheritance + a strong teacher is the only credible route, and even then sub-10M
# alignment is an open problem. We ship 11-86M now and PoC the 5M tier as future work.
STRETCH_TIERS = {
    "poc5":  ("distill MobileCLIP2-S0 -> ~5M", "weight-inherited distillation; PoC / future work"),
    "mid50": ("distill MobileCLIP-B -> ~50M",  "fills the 36M->86M gap; optional"),
}
PROMPT_TEMPLATES = ["a photo of {}", "a close-up leaf photo: {}", "a leaf with {}"]

# --- SAGE parquet-shard fetch (the working data path; NOT row-streaming) ----
# `refs/convert/parquet` has 13 shards (0000..0012); crops are scattered/clustered across them.
# Front-load shard 0 (small) + shard 8 (holds Peach); auto-stop once enough crops are covered.
SHARD_REVISION = "refs/convert/parquet"
SHARD_ORDER = [0, 8, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
MAX_SHARDS = 13
CAP_HELD_PER_CLASS = 600     # raised for the 18-crop build (ample disk); tighter CIs
CAP_TRAIN_PER_CLASS = 1000
MIN_HELD_CROPS = len(HELDOUT_CROPS)   # cover ALL held crops (some live in late shards -> full pull)
EVAL_MIN_CLASS_IMAGES = 25   # class-size floor for the held-out eval set
RESULTS_DIR = DATA_ROOT / "results"   # eval result JSONs (feed docs/paper/make_figures.py)


def canonical_crop(raw: str) -> str | None:
    """Map a raw SAGE crop string to our canonical crop name, or None if not wanted."""
    if raw is None:
        return None
    key = raw.strip().lower()
    if key in CROP_ALIASES:
        return CROP_ALIASES[key]
    # loose contains-match (handles "Corn (Maize) - field", "Citrus / Orange", etc.)
    for alias, canon in CROP_ALIASES.items():
        if alias in key:
            return canon
    return None


def safe_name(s: str) -> str:
    """Filesystem-safe class/disease name."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(s).strip())


def ensure_dirs() -> None:
    for d in (DATA_ROOT, DATASET_DIR, SPLITS_DIR, DESCRIPTORS_DIR):
        d.mkdir(parents=True, exist_ok=True)
