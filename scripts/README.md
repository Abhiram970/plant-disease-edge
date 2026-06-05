# Phase A runbook — for the RTX 4060 box

This is the exact sequence to run Phase A (data foundation) end-to-end and prove the
pipeline works on real hardware. **Total time: ~20–40 min** (most of it the small SAGE
stream). You do NOT need to download the 114 GB SAGE dataset — the script streams it.

> Full plan & rationale: `../PROJECT_GUIDE.md` (sections 2, 3, 6). Roles/Kaggle: `../CONTRIBUTING.md`.

## What runs where

| Step | Script | Hardware | Why |
|---|---|---|---|
| A1 | `build_sage_subset.py` | CPU (any box) | streams + filters SAGE; I/O-bound, no GPU |
| A2 | `build_descriptors.py` | CPU (any box) | text job; stubs without API key |
| A3 | `build_splits.py` | CPU (any box) | splits + leakage check; instant |
| **A4** | **`smoke_test.py`** | **RTX 4060 (GPU)** | **the real GPU test — loads teachers+student, 1 distill epoch, zero-shot check** |

## 0. One-time setup

```bash
# from the repo root
python -m venv .venv
.venv\Scripts\activate                 # Windows  (Linux/Mac: source .venv/bin/activate)

# install PyTorch with CUDA 12.1 FIRST (matches most 4060 setups; adjust if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# log in to HuggingFace so streaming SAGE works
huggingface-cli login                  # paste an HF token (read scope is enough)
```

Quick GPU check (should print True + your 4060):
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 1. (recommended) Probe SAGE crop names

SAGE's crop strings may not be exactly title-case. This shows what's there and whether our
11 crops match. ~1 min.
```bash
python scripts/build_sage_subset.py --probe --limit 5000
```
If any wanted crop shows `0`, add an alias in `scripts/config.py` (`CROP_ALIASES`) and re-probe.

## 2. A1 — build a SMALL subset first (test the pipeline cheaply)

```bash
python scripts/build_sage_subset.py --limit 30000 --per-class-cap 80
```
Produces `data/dataset_cleaned/...` + `data/manifest.csv`. Check the printed per-crop counts
look sane. (When you're confident, the full build is just `python scripts/build_sage_subset.py`
with no flags — but the small one is all you need for the smoke test.)

## 3. A3 — splits + leakage check

```bash
python scripts/build_splits.py
```
Must print **"leakage check PASSED — 0 shared images."** Writes `data/splits/*.csv`.

## 4. A2 — descriptors (stubs are fine for the smoke test)

```bash
python scripts/build_descriptors.py
```
Writes `descriptors/<crop>.json` for all crops. (Real source-grounded fills come later with
`ANTHROPIC_API_KEY=... python scripts/build_descriptors.py --fill` — not needed to smoke-test.)

## 5. A4 — THE GPU SMOKE TEST  ✅

```bash
python scripts/smoke_test.py
```
Expected: it loads a frozen CLIP teacher + DINOv2 + EdgeNeXt-XX-Small student, runs **1
distillation epoch** on a few hundred images, builds text prototypes, and does a **zero-shot
sanity check** on held-out crops. Finishes in a few minutes on the 4060.

**Success = the final line:** `SMOKE TEST PASSED ✔ pipeline runs end-to-end.`
(The zero-shot accuracy number will be low — it's 1 epoch on a tiny sample with stub
descriptors. We're testing the *plumbing*, not the result.)

## If something breaks — quick fixes

- `edgenext_xx_small` not found → `pip install -U timm` (need ≥ 1.0.3).
- HuggingFace 401 / auth → re-run `huggingface-cli login`.
- 0 images for a crop → run step 1 (probe), fix `CROP_ALIASES` in `config.py`.
- CUDA OOM → lower `--batch` (e.g. `python scripts/smoke_test.py --batch 16`).
- No GPU? It still runs on CPU (just slower) — the test is still valid.
- SigLIP2 isn't used in the smoke test on purpose (it needs `transformers`); we benchmark it
  later in Phase B. CLIP/OpenCLIP are enough to validate the pipeline.

## What to report back

Paste the final summary blocks from steps 2, 3, and 5 into the team chat / a GitHub issue:
- A1 per-crop counts, A3 split sizes + "leakage PASSED", A4 "SMOKE TEST PASSED".
That tells us the data foundation is solid and we can scale Phases B–C on Kaggle.
```
```
