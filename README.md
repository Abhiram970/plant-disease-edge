# Plant-Disease-Edge

A **family of compact (11–93M) vision–language models** for crop-disease diagnosis that is **trained for
high real-time accuracy on known crops** and does **zero-shot diagnosis on unseen crops** via
source-grounded symptom descriptors — deployable at $0/image on phones, laptops, and small NPUs.

> **Target venue:** *Computers and Electronics in Agriculture* — Special Issue
> **"Foundation Models in Agriculture."** Submission deadline **30 Sep 2026** (internal target Aug 2026).
>
> **Status (Jun 2026):** Phase-0 de-risk complete — the hybrid architecture is **validated on real SAGE
> data**. Full evidence trail: [`docs/paper/findings_log.md`](docs/paper/findings_log.md).

---

## The idea in one line

Frontier VLM disease agents (e.g. **SAGE**, **SCOLD**) are accurate but cloud-scale. We build a deployable
compact VLM family: a **frozen image–text backbone** + a **trained head for seen crops** (high accuracy)
+ a **descriptor-driven zero-shot head for unseen crops** + **WiSE-FT** (train without losing
generalization) + an **abstain gate** — at $0/image on the edge.

## What's validated (Phase-0)

- **A frozen compact CLIP does cross-crop zero-shot** via source-grounded descriptors (MobileCLIP2-S0, 11M).
- **Descriptor quality is the lever:** rich vs bare = **+8–15pp** (matches SAGE's +14–16pp), even at 11M.
- **The hybrid works on ONE 11M backbone:** trained head **67%** on seen crops **+** zero-shot **~27%** on
  unseen crops, simultaneously.
- **Encoder bake-off:** SigLIP2 best teacher (31.5%); lightweight tiers 22–29%; **BioCLIP2 poor (dropped)**.
- **Training a tiny model from scratch / specializing on seen does NOT improve unseen zero-shot** — so we
  keep frozen backbone + descriptors for unseen, and train only the seen head (+ WiSE-FT).

## The model family

| Class | Base (open_clip) | Image-enc params | Device |
|---|---|---|---|
| Lightweight | MobileCLIP2-S0 · MobileCLIP-S1 · MobileCLIP2-S2 | ~11 / 21 / 35M | NPU · phone · laptop |
| Heavyweight | MobileCLIP-B · SigLIP2 | ~86 / 93M | workstation |
| Stretch (PoC) | distilled ~5M (TinyVLM / CLIP-RD) | ~5M | MCU — future work |

Only the **image encoder** deploys; the text encoder runs offline to precompute descriptor prototypes.

## Data

**SAGE only** (`tirtho149/SAGE`, MIT) — parquet-shard fetch on Kaggle (never the ~133 GB). Train crops:
Tomato/Soybean/Apple/Corn/Grape/Potato/Rice. Held-out (zero-shot): Coffee/Orange/Peach/Pumpkin.
Descriptors are **source-grounded** `{value, source_url, verbatim_quote}` — auditable, hallucination-resistant.

## Run the experiments

Self-contained Kaggle script (no clone/auth needed): upload **`temp/run_all.py`**, GPU + Internet ON, then
`%run run_all.py` — runs EXP1 (encoder bake-off) + EXP2 (hybrid: train seen / keep unseen) + EXP3
(fine-tune + WiSE-FT). The consolidated, importable pipeline lives in **`scripts/`** (`evaluate.py`,
`sage_data.py`, `descriptors.py`, `zeroshot.py`). Figures: `python docs/paper/make_figures.py`.

Full plan, architecture, acceptance criteria: **[`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)**.

## Getting started (new teammate)

1. Read **[`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)** (the plan) then **[`CONTRIBUTING.md`](CONTRIBUTING.md)**
   (roles, Kaggle accounts, env, git).
2. Set up the environment (CONTRIBUTING §3) and grab your role + Kaggle account (§1–2).
3. Current state: **Phase-0 de-risk complete (hybrid validated)**; next is Phase A (clean data incl.
   Coffee via a saved Kaggle Dataset, source-grounded descriptors, full ablation + on-device benchmark).

## Repo layout

```
PROJECT_GUIDE.md   # canonical plan (read first)
CONTRIBUTING.md    # how we work: roles, Kaggle, env, git
scripts/           # consolidated pipeline: config, sage_data, descriptors, zeroshot, evaluate
temp/              # run_all.py — self-contained Kaggle experiment driver (EXP1/2/3)
descriptors/       # per-crop source-grounded symptom JSON (Phase A2)
docs/paper/        # paper.md draft, findings_log.md (evidence trail), make_figures.py, figures/
```

Weights, datasets, and embedding caches are **not** in git (see `.gitignore`) — they live on Kaggle
Datasets / Hugging Face, linked from the relevant phase.

## Data & licensing

We use the **SAGE** dataset (Arshad et al., arXiv 2605.09768; HF `tirtho149/SAGE`, MIT). We publish
only our filtered subset, embeddings, and trained weights. Underlying SAGE sub-datasets keep their own
terms — we don't rehost the full corpus. See `PROJECT_GUIDE.md §2`.

## History

Supersedes an earlier tomato-ensemble project, kept separately at
`github.com/Abhiram970/Plant-disease`. This repo is a clean slate for the SAGE-distillation paper.
