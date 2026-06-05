# Plant-Disease-Edge

A **1.3M-parameter, INT8 edge model** for crop-disease diagnosis, distilled from a frozen
vision-language foundation model. The headline capability is **cross-crop zero-shot generalization**:
the tiny student diagnoses crops it never saw during training, by matching images to LLM-authored,
source-grounded symptom **text prototypes** — a capability only the foundation-model teacher enables,
delivered on a $35 device at zero marginal inference cost.

> Target venue: *Computers and Electronics in Agriculture* — Special Issue **"Foundation Models in
> Agriculture."** Internal deadline: **August 2026.**

## Idea in one line

Frontier VLM disease agents (e.g. SAGE) are accurate but cost ~$0.30/image and can't run on a phone.
We **distill that symptom-grounded knowledge into EdgeNeXt-XX-Small**, keep zero-shot generalization
via text prototypes, add an **OOD/abstain gate**, quantize to INT8, and benchmark on **phone + Raspberry Pi**.

## Approach (see `PROJECT_GUIDE.md` for the full locked plan)

- **Data:** SAGE only (`tirtho149/SAGE`, MIT) — stream-and-filter on Kaggle, never download the 114GB.
  Train on 7 crops (Tomato, Soybean, Apple, Corn, Grape, Potato, Rice); hold out 4 crops
  (Coffee, Orange/HLB, Peach, Pumpkin) for the **zero-shot** test; SAGE long-tail = OOD/abstain set.
- **Teachers (frozen):** benchmark 3 text-aligned VLMs — CLIP ViT-B/16, OpenCLIP ViT-L/14, **SigLIP2** —
  and pick the best zero-shot distiller; **DINOv2-small** as an auxiliary vision-only robustness teacher.
- **Student:** EdgeNeXt-XX-Small (~1.3M params) with a text-projection head (the zero-shot engine) +
  a DINOv2-alignment head (auxiliary). INT8 → ONNX → TFLite.
- **Descriptors:** LLM-authored, **source-grounded** (`{value, source_url, verbatim_quote}`) — auditable,
  hallucination-resistant; these are what make zero-shot possible.

## Repo layout

```
PROJECT_GUIDE.md        # the full 10-week plan (canonical)
scripts/                # data build, descriptor generation, training, export
notebooks/              # Kaggle notebooks (stream-filter, embedding cache, distillation)
descriptors/            # per-crop source-grounded symptom JSON
docs/                   # paper assets, figures (paper.tex lives here later)
```

Models, datasets, and embedding caches are **not** in git (see `.gitignore`) — they live on Kaggle
Datasets and Hugging Face.

## Status

Planning complete; execution starting (Phase A: SAGE subset build). This repo supersedes an earlier
tomato-ensemble project (kept separately at `github.com/Abhiram970/Plant-disease`).
