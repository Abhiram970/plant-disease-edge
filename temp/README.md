# temp/ — Phase 0 staging (Kaggle scratch)

**Throwaway staging area.** Scripts here are run on Kaggle to de-risk the project
before we commit them to proper locations (`scripts/`, `notebooks/`) and proper
storage (Kaggle Datasets / HF). Nothing here is canonical yet.

---

## What's here

| File | Purpose | Where it runs |
|---|---|---|
| `phase0_spike.py` | The Phase 0 **de-risk spike** — one self-contained file, no repo imports | Kaggle GPU notebook |

## Phase 0: the one question

> Does a ~1.3M-param EdgeNeXt student, distilled from a frozen CLIP teacher on a few
> **train** crops, still do **zero-shot** diagnosis on crops it **never saw**, via text
> prototypes — above chance, and how much of the teacher's zero-shot does it **retain**?

This is the load-bearing claim of the paper. We answer it on a tiny budget BEFORE the
full Phase A–E build.

## Run it on Kaggle

1. **New Notebook** → Settings: **Accelerator = GPU T4 ×1** (or P100), **Internet = ON**.
   (Internet is required: we stream SAGE from HuggingFace and pull CLIP/EdgeNeXt weights.)
2. First cell — deps not already on the Kaggle image:
   ```python
   !pip -q install "datasets>=2.19" "open_clip_torch>=2.24" "timm>=1.0.3"
   ```
3. Upload `phase0_spike.py` (Add Data → or the notebook file panel), then:
   ```python
   %run phase0_spike.py
   ```
4. Read the **RESULT block** at the end and `/kaggle/working/phase0_result.json`.

Expected runtime: **~1–2 h** on a single T4 with the defaults. To go faster, lower
`ROW_LIMIT` and the per-class caps at the top of the script.

## Reading the result (Gate 1)

The script prints a verdict by comparing **student** zero-shot to **chance** and to the
**teacher** upper bound:

| Verdict | Trigger | Action |
|---|---|---|
| **GO** | student ≥ 3× chance **and** retains ≥ 70% of teacher | Proceed to full Phase A–E. |
| **WEAK** | beats chance but retains 45–70% | Apply Gate-1 pivots (bigger/more-diverse distill set, ~5M student, SigLIP2 teacher), re-spike. |
| **NO-GO** | not clearly above chance | Pivot the headline (few-shot cross-crop, or retention/efficiency story) before sinking 8 weeks. |

## Caveats (this is a spike, not the paper)

- **Naive class-name text prototypes** ("a photo of {disease} on {crop} leaf"), not the
  source-grounded descriptors. Source-grounding is Phase A/C; here we only need to know
  if the *mechanism* survives distillation.
- Only **2 held-out crops** and small caps — a signal, not a final number.
- Single CLIP teacher (no SigLIP2/DINOv2, no INT8). Those are later phases.
- Uses the teacher's preprocess for the student too (fine for a spike).

## After it runs

Paste the printed RESULT block / attach `phase0_result.json` back here and we'll read the
gate together and decide GO / WEAK / NO-GO. On **GO**, the spike's stream-filter and
distill code graduate into `scripts/` (parameterized) for Phase A–C.
