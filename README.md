# Plant-Disease-Edge

A **~1.3M-parameter, INT8 edge model** for crop-disease diagnosis, distilled from a frozen
vision-language foundation model. The headline capability is **cross-crop zero-shot generalization**:
the tiny student diagnoses crops it **never saw during training**, by matching images to LLM-authored,
source-grounded symptom **text prototypes** — a capability only the foundation-model teacher enables,
delivered on a **$35 device at zero marginal inference cost**.

> **Target venue:** *Computers and Electronics in Agriculture* — Special Issue
> **"Foundation Models in Agriculture."**  **Internal deadline: August 2026.**

---

## The idea in one line

Frontier VLM disease agents (e.g. **SAGE**) are accurate but cost ~**$0.30/image** and can't run on a
phone. We **distill that symptom-grounded knowledge into EdgeNeXt-XX-Small**, keep **zero-shot
generalization** via text prototypes, add an **OOD/abstain gate**, quantize to **INT8**, and benchmark
on **phone + Raspberry Pi**.

## What makes it publishable

- **Primary:** zero-shot diagnosis of **held-out crops** (Coffee, Orange/HLB, Peach, Pumpkin) the model
  never trained on — the foundation-model-native capability the SI is about.
- **Mechanism is load-bearing:** a student *without* descriptors scores at chance on held-out crops →
  the LLM-descriptor anchoring is provably what enables the result, not a nice-to-have.
- **Support:** in-the-wild accuracy on 7 trained crops · honest abstention (risk-coverage) · real
  on-device latency/energy on a $35 Pi.

## Approach at a glance

| Piece | Choice |
|---|---|
| **Data** | **SAGE only** (`tirtho149/SAGE`, MIT) — stream-and-filter on Kaggle, never download the 114 GB |
| **Train crops (7)** | Tomato, Soybean, Apple, Corn, Grape, Potato, Rice |
| **Held-out / zero-shot (4)** | Coffee, Orange (HLB), Peach, Pumpkin |
| **OOD / abstain** | SAGE long-tail crops + rare classes |
| **Teachers (frozen)** | benchmark **CLIP ViT-B/16 · OpenCLIP ViT-L/14 · SigLIP2**, pick best zero-shot distiller; **DINOv2-small** as auxiliary robustness teacher |
| **Student** | EdgeNeXt-XX-Small (~1.3M) — text-projection head (zero-shot engine) + DINOv2-align head (aux) → INT8 → ONNX → TFLite |
| **Descriptors** | LLM-authored, **source-grounded** `{value, source_url, verbatim_quote}` — auditable, hallucination-resistant |

## Roadmap (10 weeks → August)

| Phase | What | Owner |
|---|---|---|
| **A** | SAGE subset + descriptors + splits | DATA |
| **B** | Cache frozen teacher embeddings once (shared Kaggle Dataset) | MODEL |
| **C** | Distill student · **zero-shot experiment** · abstain gate · ablations | MODEL |
| **D** | INT8 + phone/Pi benchmark | EDGE |
| **E** | Paper draft + submission | WRITE/LEAD |

Full detail, acceptance criteria, and experiments matrix: **[`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)**.

## Getting started (new teammate)

1. Read **[`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)** (the plan) then **[`CONTRIBUTING.md`](CONTRIBUTING.md)**
   (roles, Kaggle accounts, env, git).
2. Set up the environment (CONTRIBUTING §3) and grab your role + Kaggle account (§1–2).
3. Pick up your phase's first task in the guide (§6). Current state: **Phase A, not yet started.**

## Repo layout

```
PROJECT_GUIDE.md   # canonical plan (read first)
CONTRIBUTING.md    # how we work: roles, Kaggle, env, git
scripts/           # data build, descriptors, training, export
notebooks/         # Kaggle notebooks
descriptors/       # per-crop source-grounded symptom JSON
docs/              # paper.tex, figures (Phase E)
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
