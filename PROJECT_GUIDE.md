# PROJECT GUIDE — Plant-Disease-Edge

**Cross-crop zero-shot crop-disease diagnosis on a $35 device, distilled from a frozen vision-language
foundation model.**

- **Target venue:** *Computers and Electronics in Agriculture* (CEA, IF ≈ 8), Special Issue
  **"Foundation Models in Agriculture."**
- **Deadlines:** hard internal **August 2026** (~10 weeks of focused work); official SI deadline 30 Sep 2026.
- **This is the canonical plan.** Everything (scope, data, architecture, phases, ownership) lives here.
  Read this top-to-bottom before writing code. See `CONTRIBUTING.md` for how we work together.

---

## Table of contents
0. [Thesis & why this gets accepted](#0-thesis--why-this-gets-accepted)
1. [Glossary (read first)](#1-glossary-read-first)
2. [The dataset: SAGE](#2-the-dataset-sage)
3. [The curated subset (LOCKED)](#3-the-curated-subset-locked)
4. [Architecture (LOCKED)](#4-architecture-locked)
5. [Compute & Kaggle execution](#5-compute--kaggle-execution)
6. [Phases A–E (detailed, with owners & acceptance criteria)](#6-phases-ae)
7. [Experiments & results matrix](#7-experiments--results-matrix)
8. [Scope: in / deferred / never](#8-scope-in--deferred--never)
9. [Repo layout & conventions](#9-repo-layout--conventions)
10. [Risks & mitigations](#10-risks--mitigations)
11. [Literature anchors](#11-literature-anchors)

---

## 0. Thesis & why this gets accepted

Frontier vision-language models (VLMs) — e.g. the **SAGE** agentic diagnoser — reach high in-the-wild
disease accuracy but cost **~$0.21–0.42 per image** and **cannot run on a phone or a Raspberry Pi**.
We **distill that foundation-model symptom knowledge into a ~1.3M-parameter INT8 EdgeNeXt-XX-Small
student** that runs on a $35 device at zero marginal inference cost.

**PRIMARY HEADLINE — cross-crop zero-shot generalization.** The tiny student, anchored on
LLM-authored symptom-descriptor *text prototypes*, diagnoses crops it **never saw during training**
(zero-shot, by matching an image to descriptor text). This is a capability that only a
foundation-model distillation enables, and it is the open problem (Ghazal et al. 2024: "models trained
on one crop don't transfer to others") that the field has not solved at edge scale.

**SUPPORTING results:** (a) in-the-wild accuracy on the trained crops; (b) an **OOD / abstain gate**
for honest "I don't know" field behavior; (c) a real **phone + Raspberry Pi** efficiency benchmark.

**Why this framing maximizes acceptance at THIS Special Issue:** the SI theme is *Foundation Models
in Agriculture.* Zero-shot transfer to unseen crops via text prototypes is **foundation-model-native**
— impossible without the CLIP/VLM teacher — so it answers the theme head-on, unlike a generic "small
CNN, good accuracy" paper (which would read as 2021 work). Generalization is the lead claim; edge
efficiency and abstention are the support. The journal has accepted closely related work
(WDLM, the VLM-for-ag canon below), so the framing is on-target for the editors.

**One-line positioning vs SAGE:** SAGE is an accurate but ~$0.30/image **cloud** agent that needs
reference images per crop; we are the deployable **edge** student that generalizes to unseen crops via
text prototypes at $0/image.

---

## 1. Glossary (read first)

| Term | Meaning in this project |
|---|---|
| **Teacher** | A large, **frozen** pretrained model we distill knowledge *from*. We never train it. |
| **Student** | The small (~1.3M-param) model we actually train and ship to the edge device. |
| **Distillation** | Training the student so its features/predictions match the teacher's. |
| **Text-aligned (image–text) space** | An embedding space where an image vector and a text vector are comparable (cosine similarity). CLIP/OpenCLIP/SigLIP2 have this. **DINOv2 does not.** |
| **Text prototype** | The embedding of a disease's *symptom-descriptor text*. We classify an image by finding its nearest text prototype. |
| **Zero-shot** | Diagnosing a crop/disease with **no training images** for it — only its text prototype. The headline capability. |
| **Descriptor** | A structured, source-grounded symptom description per disease: `{value, source_url, verbatim_quote}`. LLM-authored, expert-audited. |
| **OOD / abstain gate** | Logic that outputs "unknown / abstain" when the image is far from every known prototype (e.g. a crop we don't cover). |
| **OOD set** | Out-of-distribution images used to test abstention — here, SAGE's long-tail crops + rare classes. |
| **Embedding cache** | Teacher embeddings precomputed once and stored, so student training never re-runs the teacher. |
| **SAGE** | The plant-disease image+symptom dataset we use (see §2). |
| **PTQ / QAT** | Post-Training Quantization / Quantization-Aware Training — two ways to make the model INT8. |

---

## 2. The dataset: SAGE

We use **SAGE only.** No PlantVillage, no PlantDoc, no PlantWild (lab-only/saturated/2016-era — using
them would invite "old, lab-only" reviews and there's no need).

**Official sources (confirmed):**
- **Images (Hugging Face):** `https://huggingface.co/datasets/tirtho149/SAGE`
  — **~1.01M rows / ~114 GB, MIT license**, stored as **Parquet shards**, ONE flat `train` split,
  columns `{image: bytes, crop: str, disease: str, filename: str}`. **NOT per-crop downloadable.**
- **Code + symptom registry KB:** `https://github.com/tirtho149/SAGE` (the per-disease symptom
  knowledge base + `open_agentic/prompt.py` live here — this is the descriptor goldmine).
- Project page: `https://sage-dataset.github.io/` · Paper: arXiv 2605.09768.

**License note:** the HF aggregate is **MIT** → fine to train/evaluate/cite a research model on. The
underlying sub-datasets keep their own terms (a couple were NC-ND in the paper's appendix), so do
**not** rehost the whole corpus or claim it as ours; for any redistributed derivative, prefer the
clearly-permissive crops. We only ever publish *our filtered subset + our embeddings/weights.*

**Why we don't take all 114 GB:** off-strategy (we want a focused, deployable model, not maximal
coverage) **and** infeasible on Kaggle's storage + 9-hour limits.

**The pull strategy — STREAM + FILTER (never download 114 GB):** because it's a HF `datasets` repo we
stream rows and keep only our crops up to a per-class cap. Only the *kept* images (~6–8 GB) ever land
on disk.
```python
from datasets import load_dataset
ds = load_dataset("tirtho149/SAGE", streaming=True, split="train")  # NO full download
TRAIN   = {"Tomato","Soybean","Apple","Corn","Grape","Potato","Rice"}  # 7 trained crops
HELDOUT = {"Coffee","Orange","Peach","Pumpkin"}                        # 4 zero-shot test crops
WANT    = TRAIN | HELDOUT
# iterate rows → keep row["crop"] in WANT → cap per (crop,disease) → decode bytes → save jpg + manifest
```
**Run this ON Kaggle** (fast HF bandwidth; the output becomes a Kaggle Dataset directly, which feeds
the embedding-cache step; keeps the firehose off your home connection).

---

## 3. The curated subset (LOCKED)

Chosen from the thick, well-populated wedges of the SAGE distribution (~839K labeled images).

### TRAIN / DISTILL crops (7) — broad grower base, "foundational" coverage
**Tomato, Soybean, Apple, Corn (Maize), Grape, Potato, Rice.**
Representative dense classes (from the SAGE sunburst):
- **Tomato:** Leaf Mosaic Virus, Target Spot, Leaf Mold, Early Blight, Bacterial Leaf Spot, Late
  Blight, Septoria Leaf Blotch, Tomato Leaf Curl Virus.
- **Soybean:** Frogeye Leaf Spot, Bacterial Blight, Brown Spot, Bacterial Pustule, + deficiencies
  (Iron Chlorosis, Potassium), Herbicide Injury.
- **Apple:** Cedar Apple Rust, Black Rot, Alternaria Blotch, Apple Mosaic Virus, Apple Scab, Grey Spot.
- **Corn:** Common Rust, Northern Leaf Blight, Maize Streak Virus, Gray Leaf Spot.
- **Grape:** Black Measles, Black Rot, Leaf Blight.
- **Potato:** Late Blight, Alternaria, Bacterial Leaf Spot.
- **Rice:** Bacterial Leaf Spot, Blast, Tungro, Brown Spot.

**Tomato = showcase crop** (highest inter-class similarity → the case where descriptor anchoring should
beat a plain student; mirrors SAGE's own large Tomato gain from adding symptom knowledge).

### HELD-OUT ZERO-SHOT crops (4) — never trained — the PRIMARY result
**Coffee, Orange (Huanglongbing), Peach, Pumpkin.** Distinctive, well-populated classes. The student
classifies these using **only** the LLM-descriptor text prototypes (zero training images). This table
is the headline of the paper.

### OOD / ABSTAIN set
The SAGE long-tail ("+315 more" crops) + any class with **< 50 images**. The abstain gate must say
"unknown" on these. No external weed/insect datasets needed — SAGE's own long tail is the OOD source.

### Budgets & splits (LOCKED defaults)
- **Per-class cap:** ~**1,500 images/class**. Drop classes < 50 images → OOD set.
- **Target totals:** ~**80–110 classes**, ~**6–8 GB** images, ~**3–4 GB** embedding cache → fits Kaggle.
- **Within trained crops:** random **80/10/10** train/val/test per class.
- **"Harder-conditions" slice:** where the manifest exposes image source, hold the in-the-wild-looking
  images as an extra test slice (recovers a difficulty axis without external datasets).
- **Leakage rule:** verify by content-hash that NO image appears in more than one split role.

### Descriptors (the zero-shot fuel)
Seed from the SAGE GitHub registry for **all 11 crops** (held-out included — descriptors are *how*
zero-shot works), then extend/normalize with Claude using SAGE's **source-grounded schema**: every
field is `{value, source_url, verbatim_quote}`, the LLM is **forbidden from using its own knowledge**,
and a sample is expert spot-audited. This makes the headline novelty auditable and pre-empts the
"LLM hallucination" reviewer critique.

---

## 4. Architecture (LOCKED)

**Critical design fact:** zero-shot to UNSEEN crops requires an **image–text aligned** space.
CLIP-family teachers have it; **DINOv2 is vision-only and CANNOT do zero-shot.** Therefore the headline
rests on the CLIP-family alignment, and **DINOv2 is an auxiliary teacher for trained-crop robustness
only.** State this explicitly in the paper — it pre-empts the obvious reviewer question.

### Teachers (all FROZEN — never trained)
- **Text-aligned (the zero-shot engine) — we BENCHMARK 3 and pick the winner (Phase C0):**
  - **CLIP ViT-B/16** — 512-d, smallest cache, baseline.
  - **OpenCLIP ViT-L/14 (LAION-2B)** — 768-d, strong, zero availability risk.
  - **SigLIP2** (2025) — SOTA open zero-shot. **Verify weights load on Kaggle/timm in Phase A.**
  - The comparison *"which foundation model distills best into an edge student for zero-shot?"* is
    itself a paper contribution.
- **Auxiliary (trained crops only):** **DINOv2-small** — vision-only structure/robustness signal.

### Student — EdgeNeXt-XX-Small (~1.3M params), TWO heads
1. **Text-projection head** → projects the student's image features into the winning teacher's
   text-aligned space. **THIS IS THE ZERO-SHOT ENGINE:** descriptors → text prototypes → cosine
   nearest-neighbor classifies any crop, seen or unseen. *A student without descriptors has no
   prototypes → 0% zero-shot → the descriptor anchoring is provably the load-bearing component.*
2. **DINOv2-alignment head** (auxiliary) → regularizes visual structure; improves trained-crop and
   "harder-conditions" accuracy. Droppable at inference.

**Fallback if 1.3M underfits ~110 classes:** MobileNetV4-Conv-S (~2–3M) or TinyViT-5M. Decide
empirically in Phase C — do **not** pre-commit to a struggling backbone.

**Story to tell:** text-aligned FM → generalization; DINOv2 → robustness; distillation → a 1.3M INT8
model that has both, on a $35 device. Clean three-way ablation: text-teacher-only vs +DINOv2 vs neither.

---

## 5. Compute & Kaggle execution

- **Hardware:** 1× RTX 4060 (a teammate runs first / smoke tests) + **3 Kaggle accounts**
  (~90 GPU-hr/week total) + Colab credits as overflow.
- **Kaggle background runs:** use **"Save Version → Run All (commit)"** — survives tab close, 9-hour
  cap, and avoids the 40-minute idle-timeout death. **Every run must checkpoint + resume so it fits
  inside 9 hours.**
- **The key trick that makes this feasible:** compute the **frozen teacher embeddings ONCE**, publish
  them as a **shared Kaggle Dataset** that all 3 accounts read. Student training then needs only the
  cached vectors + small images → each student run drops to ~3–5 hours. **The few-GB embedding cache
  is the thing that must fit — never the 114 GB.**
- **Account split:** see `CONTRIBUTING.md` (each member owns one Kaggle account; we coordinate so we
  don't double-spend GPU hours on the same run).

---

## 6. Phases A–E

> Owners use role tags: **DATA**, **MODEL**, **EDGE**, **WRITE**, **LEAD**. Assign real names in
> `CONTRIBUTING.md`. Each task lists a **Deliverable** and **Done when** (acceptance criteria).

### Phase A — Data foundation (Week 1–2) · owner: DATA
- **A1** — `scripts/build_sage_subset.py` (run as a Kaggle notebook). Stream `tirtho149/SAGE`, filter
  `crop ∈ TRAIN∪HELDOUT` (11 crops), cap per (crop,disease) ~1,500, decode image bytes →
  `dataset_cleaned/<Crop>___<Disease>/<filename>`, content-hash dedupe, drop classes < 50 imgs (→ OOD
  list). Emit `manifest.csv` with columns `path, crop, disease, filename, split_role`
  (`split_role ∈ {train_crop, heldout_crop, ood}`). Publish result as a Kaggle Dataset.
  **Deliverable:** the script + a published Kaggle Dataset + `manifest.csv`.
  **Done when:** subset is ~6–8 GB, ~80–110 classes, the full 114 GB was never downloaded, and a second
  Kaggle account can load the published dataset.
- **A2** — `scripts/build_descriptors.py`. Clone `tirtho149/SAGE`, pull the symptom registry for all 11
  crops as a seed, extend/normalize with Claude using the source-grounded schema → `descriptors/<crop>.json`.
  **Deliverable:** 11 descriptor JSONs, each disease carrying `{value, source_url, verbatim_quote}`.
  **Done when:** every held-out-crop disease has a descriptor (no gaps), and a spot-audited sample
  (≥ 1 crop) shows quotes that actually support the claims.
- **A3** — Build splits from the manifest alone (no external data): trained-crop 80/10/10 per class;
  held-out crops reserved entirely for zero-shot; long-tail + dropped classes = OOD.
  **Deliverable:** `splits/{train,val,test,heldout,ood}.csv`.
  **Done when:** a hash check proves **zero image leakage** across roles.
- **A4** — Smoke test on the RTX 4060: full data build + 1-epoch student run end-to-end.
  **Done when:** it runs start→finish with no path/format errors and produces a loss curve.

### Phase B — Teacher embedding cache (Week 2–3) · owner: MODEL (+ DATA for publishing)
- **B1** — Kaggle notebook: embed all subset images ONCE with the **3 frozen text teachers**
  (CLIP-B/16, OpenCLIP-L/14, SigLIP2) **+ DINOv2-small** (aux). Compute the matching **text-prototype**
  embeddings from the descriptors **for each text teacher**. Checkpoint per shard (< 9 hr). This is
  ~3× *caching*, not 3× *training* (teachers are frozen).
  **Deliverable:** `embeddings/<teacher>/*.npy` + `prototypes/<teacher>.npy`.
  **Done when:** every subset image has an embedding from all 4 models, and every disease has a text
  prototype per text-teacher.
- **B2** — Publish embeddings + per-teacher prototypes as a shared Kaggle Dataset.
  **Done when:** it loads cleanly from a second account; this single asset feeds every student run.

### Phase C — Distillation student (Week 3–6) · owner: MODEL
- **C0 — Teacher bake-off.** Train the lightweight student head against each cached text teacher;
  measure zero-shot held-out accuracy; pick the winner as primary teacher. (DINOv2 stays aux.)
  **Deliverable:** a 3-row table (CLIP / OpenCLIP / SigLIP2 → zero-shot acc) → goes in the paper.
- **C1 — Student training.** EdgeNeXt-XX-Small, two heads (§4). **3-stage curriculum:**
  (i) feature alignment to teacher embeddings (winning text teacher + DINOv2 aux);
  (ii) **text-prototype anchoring** on LLM descriptors (this is what enables zero-shot);
  (iii) fine-tune on trained crops. Fallback backbone if it underfits.
  **Done when:** trained-crop test accuracy is competitive and the run fits the 9-hr Kaggle window.
- **C2 — PRIMARY EXPERIMENT: zero-shot held-out crops.** Classify the 4 unseen crops by nearest
  descriptor text prototype (no training images). Compare against (a) the SAGE cloud agent's reported
  accuracy and (b) a **no-descriptor student** (which is structurally incapable of zero-shot).
  **Deliverable:** the headline results table.
  **Done when:** zero-shot accuracy on ≥ 4 held-out crops is reported with the no-descriptor baseline at 0/chance.
- **C3 — OOD / abstain gate.** Threshold on distance to nearest prototype; calibrate on a trained-crop
  val split; evaluate abstention on the OOD set; report an **accuracy-vs-coverage (risk-coverage) curve**.
- **C4 — Ablations.** no-descriptor vs descriptor (does it enable/boost zero-shot + mirror SAGE's
  +16.2 pp on trained crops at edge scale?); single- vs dual-teacher; naive-prompt vs source-grounded
  descriptor.

### Phase D — Edge optimization + on-device benchmark (Week 6–8) · owner: EDGE
- **D1** — INT8 quantization (PTQ first; QAT if accuracy drops too far). Export ONNX → TFLite.
  **Done when:** INT8 model runs and matches FP accuracy within an acceptable margin (record the delta).
- **D2** — Real-device benchmark on **BOTH** Android (TFLite) **and** Raspberry Pi: params, GFLOPs,
  INT8 size, **p50/p95 latency, RAM, energy/inference**. Frame tiny-model efficiency *as a contribution*
  (precedent: a 1.98M-param model with CPU-only FPS was accepted as a result in this venue family).
  **Deliverable:** an efficiency table + on-device latency plot.

### Phase E — Writing & submission (Week 8–10) · owner: WRITE + LEAD
- **E1** — New `docs/paper.tex`. Reuse SAGE's dataset-comparison table; **lead the results with the
  zero-shot held-out-crop table** (the headline).
- **E2** — Rigor checklist: sign test / Wilcoxon for student-vs-baseline; Grad-CAM panels (incl. under
  blur/illumination/occlusion corruption — cheap robustness curve, reconsider un-cutting if time);
  full metric set (P/R/F1, per-crop accuracy, risk-coverage curve, on-device latency).
- **E3** — Positioning + citations: SAGE = accurate-but-$0.30/img cloud agent needing per-crop
  reference images; **us = deployable INT8 edge distillation that generalizes to unseen crops via text
  prototypes at $0/img.** Cite the VLM-for-ag canon (§11). Frame zero-shot cross-crop transfer as the
  SI's "foundation model" contribution.
  **Done when:** the draft is complete, all tables/figures populated, and co-authors have signed off.

---

## 7. Experiments & results matrix

| # | Experiment | Role | Headline? | Goes in paper as |
|---|---|---|---|---|
| E-ZS | Zero-shot accuracy on 4 held-out crops (vs no-descriptor=chance, vs SAGE cloud) | MODEL | **YES** | Main results table |
| E-TB | Teacher bake-off: CLIP-B/16 vs OpenCLIP-L/14 vs SigLIP2 | MODEL | secondary contribution | Ablation table |
| E-IW | In-the-wild accuracy on the 7 trained crops (+ harder-conditions slice) | MODEL | support | Per-crop table |
| E-OOD | Abstain gate: risk-coverage curve on OOD set | MODEL | support | Risk-coverage figure |
| E-ABL | Ablations: ±descriptor, ±DINOv2, naive-prompt vs grounded | MODEL | support | Ablation table |
| E-EDGE | INT8 size + p50/p95 latency + RAM + energy on phone & Pi | EDGE | support (key for SI) | Efficiency table + plot |
| E-XAI | Grad-CAM panels (clean + corrupted) | WRITE | optional | Qualitative figure |

---

## 8. Scope: in / deferred / never

**IN (August):** LLM source-grounded descriptor distillation · text-aligned teacher (best of 3) +
DINOv2 aux · zero-shot held-out crops · OOD/abstain gate · INT8 · phone + Pi benchmark.

**DEFERRED (mid-2027 / extension):** severity estimation (Disease Severity Index) · full robustness
corruption benchmark (adopt the cheap OpenCV version only if time allows) · fruit/multi-organ
extension · few-shot cassava · a polished deployed farmer app (we'll build a fresh demo *after* the
core results, not now).

**NEVER (policy):** PlantVillage, PlantDoc, PlantWild (SAGE-only) · synthetic-augmentation as a data
source (reviewers discount it) · the old tomato-ensemble weights as a contribution (heavy-baseline
comparison only, if at all).

---

## 9. Repo layout & conventions

```
PROJECT_GUIDE.md      # this file — canonical plan
README.md             # project summary + onboarding
CONTRIBUTING.md       # how we work (roles, Kaggle accounts, git, env)
scripts/              # python: data build, descriptors, training, export
  consolidate_all_data.py   # reused helper (parameterize hardcoded paths before use)
notebooks/            # Kaggle notebooks (stream-filter, embedding cache, distillation)
descriptors/          # per-crop source-grounded symptom JSON (output of A2)
docs/                 # paper.tex, figures, tables (Phase E)
```
**Not in git** (see `.gitignore`): weights (`*.pth/*.onnx/*.tflite`), data (`data/`,
`dataset_cleaned/`, `*.parquet`), embedding caches, secrets (`kaggle.json`, `.env`). These live on
**Kaggle Datasets / Hugging Face**, linked from the relevant phase.

**Conventions:** one script = one job; every script takes args, no hardcoded absolute paths; every
Kaggle notebook checkpoints; commit messages explain *why*; large artifacts go to Kaggle, links go in
the README/issue.

---

## 10. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| SigLIP2 weights don't load cleanly on Kaggle | med | Verify in Phase A; CLIP-B/16 + OpenCLIP-L/14 are zero-risk fallbacks; bake-off design means we lose nothing. |
| 1.3M student underfits ~110 classes | med | Fallback to MobileNetV4-S / TinyViT-5M; decide empirically in C1. |
| Zero-shot accuracy on held-out crops is weak | med | It's still a positive result vs no-descriptor=chance; richer/better-grounded descriptors + stronger teacher are the levers; the *mechanism* claim stands regardless. |
| Kaggle 9-hr / storage limits | high | Embedding-cache-once trick; checkpoint+resume; subset is small by design. |
| Descriptor hallucination critique | med | Source-grounded `{value,source_url,verbatim_quote}` schema + expert spot-audit. |
| Timeline slips past August | med | Phases are ordered so the headline (A→B→C2) lands first; D/E can compress; deferred list absorbs cuts. |

---

## 11. Literature anchors

- **SAGE** (Arshad et al., arXiv 2605.09768): our data + the cloud agent we contrast against;
  source for the +16.2 pp "symptom knowledge helps" result and the dataset-comparison table.
- **Ghazal et al. 2024** (CV in smart agriculture survey): states the open problem — models don't
  transfer across crops, and edge deployment needs compression/quantization. Our gap statement.
- **VLM-for-ag canon to cite:** AgroGPT, Agri-LLaVA, AgReason, ChatLeafDisease, WDLM (CEA!), PDD-Agent,
  AgMMU; plus PepperDet (text descriptions → +9% detection) as external validation that text anchoring
  helps disease vision.
- **Edge precedent:** a ~1.98M-param model benchmarked CPU-only was accepted in this venue family —
  precedent that "tiny model + real on-device latency" counts as a contribution.
