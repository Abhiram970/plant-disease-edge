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
We build **a compact, deployable model family (11 / 21 / 35M lightweight + 86 / 93M heavyweight)** that is
**trained for high real-time accuracy on known crops** and does **zero-shot diagnosis on unseen crops** via
source-grounded symptom descriptors — at zero marginal inference cost on phones, laptops, and small NPUs.
(A ~5M tier is a proof-of-concept via weight-inherited distillation; sub-10M aligned models aren't
off-the-shelf.)

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

**Positioning vs the 2025 agricultural foundation models (SCOLD, BioCLIP 2):** these are accurate but
cloud-scale VLMs trained with image–caption contrastive learning; we *distill them* into deployable
5–20M edge students, add **source-grounded** (auditable) descriptors, and prove **cross-crop** zero-shot
to unseen crops. SCOLD is therefore both a candidate **teacher** and the **baseline we beat on
deployability + cross-crop transfer** — not a competitor we ignore.

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

### Generalization axis & defensibility (answers the #1 reviewer attack)
We claim **cross-crop (unseen-host)** transfer mediated by descriptors — *not* "unseen disease." State this
explicitly so the reviewer can't redefine it. To pre-empt "you cherry-picked the held-out crops":
- **Stratify held-out results by pathogen novelty:** *familiar-type* (Coffee Leaf Rust — rust morphology
  also seen in Corn/Apple) vs *novel-type* (Citrus HLB — distinctive, unseen). Reporting both is far
  stronger than one blended number and kills "you just memorized rust."
- **Leave-one-crop-out (LOCO):** rotate which crops are held out; show the result is stable, not specific
  to one lucky split. (Phase C — the single best answer to "cherry-picked.")
- **Lead with economic impact:** Coffee Leaf Rust and Citrus HLB are globally catastrophic, visually
  distinctive diseases — "diagnosed without ever training on coffee or citrus" is the editor-facing story.
- **Own the scoping:** 11 of SAGE's 300+ crops = a focused deployable model + statistical validity
  (≥50 imgs/class), not maximal coverage.

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

### Teachers (all FROZEN — never trained) — bake-off picks the winner (Phase C0)
These double as the **zero-shot prototype encoder** AND the **distillation source** for the trained heads.
We bake them off — **only SigLIP2 is CONFIRMED so far (~25.6%); the rest are UNVERIFIED (loaders pending):**
- **SCOLD** (2025) — domain **leaf-disease** VLM (Swin-T ~28M + RoBERTa; HF `enalis/scold`, loads via
  `transformers AutoModel`, **NOT open_clip**). Reported to beat CLIP-L/BioCLIP/SigLIP2 on leaf disease.
  Test BOTH as a teacher AND as a deployable ~28M domain tier. Our closest competitor.
- **AgriCLIP** (2024, arXiv 2410.01407) — agriculture/livestock CLIP (600K image–text pairs); domain teacher.
- **BioCLIP 2** (2025) — biological foundation model (TreeOfLife-200M); load via open_clip hf-hub.
- **SigLIP2** (2025) — best **generic** open zero-shot VLM (the one we've confirmed).
- **CLIP ViT-B/16** — baseline floor / smallest cache.
- (Optional 5th: **MobileCLIP2**, low-latency teacher.)
- The comparison *"which foundation model — generic vs biological vs domain-specific — distills best into
  an edge student for cross-crop zero-shot?"* is itself a paper contribution.
- **Auxiliary (trained crops only):** **DINOv2-small** — vision-only structure/robustness signal (no
  zero-shot; droppable at inference).

### Student family — FROZEN backbone + TRAINED seen head + ZERO-SHOT unseen (REVISED 2026-06-28)

> **Phase-0 findings (Jun 2026):** (1) training a tiny model to *learn* image–text alignment from ~10K
> images fails — from-scratch ≈ chance; specializing a pretrained model on seen crops FORGETS (−4.5pp on
> unseen). A small model must INHERIT alignment from pretraining. (2) A frozen pretrained small CLIP
> already does cross-crop zero-shot (**MobileCLIP2-S0 ~11M → 19.9%**, 3.4× chance, ~78% of 93M SigLIP2).
> (3) Descriptor QUALITY is the lever (**rich +8..+15pp over bare**, matching SAGE's +14–16pp), even at
> 11M. (4) Accuracy is flat 11M→300M (efficiency pillar). **Corollary:** zero-shot is the right tool for
> UNSEEN crops; for SEEN crops, SUPERVISED training beats zero-shot (lit: VLM zero-shot ≪ supervised,
> ~62% best) — so we TRAIN thoroughly for seen crops and keep zero-shot for unseen.

**The deployable system per tier = ONE frozen backbone, two heads, a router:**
1. **Trained seen-crop head** — supervised / fine-tuned on the SAGE trained crops, trained THOROUGHLY
   with checkpoints to be the BEST real-time model on known crops. (This is where training lives.)
2. **Zero-shot descriptor head** — image embedding vs source-grounded descriptor prototypes; diagnoses
   UNSEEN crops with no training data ($0/crop). The novel cross-crop capability (headline).
3. **WiSE-FT weight ensembling** — interpolate fine-tuned ⊕ frozen weights so seen-crop gains do NOT
   destroy unseen-crop zero-shot (Wortsman et al.) — "train but keep generalization."
4. **Abstain / OOD router** — distance to nearest prototype: confident known crop → trained head; far /
   unknown → zero-shot or "unsure." Honest field behaviour.

**The family (deploy the image encoder; text encoder runs offline for prototypes):**

| Class | Base (open_clip) | Image-enc params | Device |
|---|---|---|---|
| Lightweight | MobileCLIP2-S0 | ~11.4M | small NPU / phone |
| Lightweight | MobileCLIP-S1 | ~21.5M | laptop CPU |
| Lightweight | MobileCLIP2-S2 | ~35.8M | laptop |
| Heavyweight | MobileCLIP-B | ~86.3M | workstation |
| Heavyweight / ref | SigLIP2 ViT-B/16 | ~92.9M | workstation / ceiling |
| *Stretch (PoC)* | *distilled ~5M (TinyVLM Matryoshka + CLIP-RD)* | *~5M* | *MCU — future work* |

Goal: the **best deployable real-time model at each tier** — seen-crop accuracy via thorough training,
unseen via zero-shot. (SCOLD's Swin-T ~28M is itself a candidate domain-specialized tier — test it.)

### Method (REVISED, LOCKED)
- **Seen crops:** train thoroughly (supervised head / fine-tune) on SAGE trained crops, checkpointed;
  protect unseen zero-shot with **WiSE-FT**. Goal = best per-tier real-time accuracy on known crops.
- **Unseen crops:** frozen backbone + **source-grounded descriptor prototypes** (the headline). rich >
  crude > bare is validated; Phase A2 builds auditable `{value, source_url, verbatim_quote}` descriptors.
- **Encoder / teacher bake-off:** SCOLD (`transformers` custom load) · BioCLIP2 (hf-hub) · AgriCLIP vs
  SigLIP2 (only SigLIP2 CONFIRMED) — pick the best zero-shot encoder / distill source.
- **5M PoC:** weight-inherited distillation (TinyVLM Matryoshka + CLIP-RD relational KD) from the 11M
  tier — framed as PoC + the "sub-10M aligned-model gap" contribution, not a delivered tier.

**Story:** a deployable compact family (11–93M) that is **accurate on known crops** (trained, real-time)
and **generalizes to unknown crops** via source-grounded descriptor zero-shot, with an abstain gate — at
$0/image on edge. The flat 11M→300M curve shows the small flavour is ~90% of a giant. Ablations:
±training, WiSE-FT vs naive fine-tune, ±source-grounding, encoder bake-off, across tiers.

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
- **A5** — `scripts/check_teachers.py`: load-check ALL teachers (SCOLD, BioCLIP 2, SigLIP2, CLIP) and ALL
  student backbones (EMOv2-5M, iFormer-M/L) on Kaggle/timm/HF. **Availability is the #1 risk — fail fast.**
  **Done when:** every teacher returns image+text embeddings and every backbone instantiates on Kaggle.

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
- **C1 — Student training (3 tiers).** Train EMOv2-5M, iFormer-M, iFormer-L — two heads each (§4), same
  MobileCLIP2/CLIP-KD recipe. **3-stage curriculum:** (i) feature alignment to the winning teacher
  (+DINOv2 aux); (ii) **source-grounded descriptor prototype anchoring** (enables zero-shot);
  (iii) fine-tune on trained crops.
  **Done when:** each tier's trained-crop accuracy is competitive and every run fits the 9-hr Kaggle window.
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
- **D1** — INT8 quantization (PTQ first; QAT if accuracy drops too far), **per tier**. Export ONNX → TFLite.
  **Done when:** each INT8 tier runs and matches FP accuracy within an acceptable margin (record the delta).
- **D2** — Real-device benchmark, **each tier on its target device** (EMOv2-5M → NPU smart-cam/Pi;
  iFormer-M → phone/TFLite; iFormer-L → laptop CPU): params, GFLOPs, INT8 size, **p50/p95 latency, RAM,
  energy/inference** → an **accuracy↔params↔latency Pareto plot**. Frame the scaling sweep *as a
  contribution* (precedent: a 1.98M-param CPU-only model was accepted in this venue family).
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
| E-TB | Teacher bake-off: SCOLD vs BioCLIP 2 vs SigLIP2 vs CLIP-B/16 (domain vs biological vs generic) | MODEL | secondary contribution | Ablation table |
| E-IW | In-the-wild accuracy on the 7 trained crops (+ harder-conditions slice) | MODEL | support | Per-crop table |
| E-OOD | Abstain gate: risk-coverage curve on OOD set | MODEL | support | Risk-coverage figure |
| E-ABL | Ablations: ±descriptor, ±DINOv2, naive-prompt vs grounded | MODEL | support | Ablation table |
| E-EDGE | Pareto sweep: INT8 size + p50/p95 latency + RAM + energy for 5M/10M/20M on cam·Pi / phone / laptop | EDGE | support (key for SI) | Efficiency table + Pareto plot |
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
- **2025 agricultural / bio foundation models (teachers + positioning):**
  - **SCOLD** (Soft-target COntrastive learning for Leaf Disease, arXiv 2505.07019, 2025) — domain
    leaf-disease VLM; our primary teacher candidate AND the cloud baseline we beat on edge + cross-crop.
  - **BioCLIP 2** (arXiv 2505.23883, 2025) — TreeOfLife-200M biological foundation model.
  - **MobileCLIP2** (Apple, arXiv 2508.20691, 2025) — SOTA CLIP→edge via multi-modal reinforced training
    (our distillation recipe).
- **Distillation method anchors:** **CLIP-KD** (CVPR 2024) · **TinyCLIP** (ICCV 2023, affinity mimicking +
  weight inheritance) · **ComKD-CLIP** (2024).
- **Descriptor-classification lineage (MUST cite):** **DCLIP** "Visual Classification via Description from
  LLMs" (Menon & Vondrick, ICLR 2023) · **CuPL** "Customized Prompts via Language models" (ICLR 2023).
  Our novelty over them = *source-grounded* (auditable, anti-hallucination) + *distilled to edge* + *cross-crop ag*.
- **Edge backbones (students):** **EMOv2** (5M frontier, arXiv 2412.06674) · **iFormer** (mobile hybrid,
  arXiv 2501.15369).
- **Agriculture eval / framing:** **AgroBench** (agriculture VLM benchmark, 2025) for an external number;
  **Dong Chen et al., *Foundation models in smart agriculture: Basics, opportunities, and challenges*** (CEA
  2024) — **guest editor's own survey; align the gap statement to its taxonomy.** SI deadline **30 Sep 2026**.
