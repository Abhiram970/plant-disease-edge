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
