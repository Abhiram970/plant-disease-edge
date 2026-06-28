"""
Shared configuration for Plant-Disease-Edge (Phase A and beyond).

Single source of truth for crop lists, paths, and dataset coordinates.
Import this everywhere instead of hardcoding. See PROJECT_GUIDE.md sections 2-3.
"""
from __future__ import annotations
import os
from pathlib import Path

# --- Dataset coordinates (SAGE) ---------------------------------------------
SAGE_HF_REPO = "tirtho149/SAGE"          # HuggingFace datasets repo (images, MIT, ~114GB parquet)
SAGE_GH_REPO = "https://github.com/tirtho149/SAGE"  # symptom registry / KB lives here

# --- Crop split (LOCKED, PROJECT_GUIDE.md sec 3) ----------------------------
# 7 crops we TRAIN/DISTILL on (broad grower base).
TRAIN_CROPS = ["Tomato", "Soybean", "Apple", "Corn", "Grape", "Potato", "Rice"]
# 4 crops we hold out entirely for the ZERO-SHOT headline result (never trained).
HELDOUT_CROPS = ["Coffee", "Orange", "Peach", "Pumpkin"]
WANT_CROPS = TRAIN_CROPS + HELDOUT_CROPS  # 11 crops total

# SAGE crop strings may differ in spelling/case (e.g. "Corn (Maize)", "corn",
# "Orange/Citrus"). We match case-insensitively on these aliases -> canonical name.
CROP_ALIASES = {
    "tomato": "Tomato",
    "soybean": "Soybean", "soya": "Soybean", "soya bean": "Soybean",
    "apple": "Apple",
    "corn": "Corn", "maize": "Corn", "corn (maize)": "Corn",
    "grape": "Grape", "grapevine": "Grape",
    "potato": "Potato",
    "rice": "Rice",
    "coffee": "Coffee",
    "orange": "Orange", "citrus": "Orange", "sweet orange": "Orange",
    "peach": "Peach",
    "pumpkin": "Pumpkin",
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

DATASET_DIR = DATA_ROOT / "dataset_cleaned"   # <Crop>___<Disease>/<filename>.jpg
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
CAP_HELD_PER_CLASS = 120
CAP_TRAIN_PER_CLASS = 250
MIN_HELD_CROPS = 2            # stop once >=N held crops are covered (Coffee is rare in early shards)
EVAL_MIN_CLASS_IMAGES = 15   # class-size floor for the held-out eval set
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
