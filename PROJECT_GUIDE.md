# PROJECT GUIDE 2 — CEA Distillation Paper (SAGE-subset edition)

> Supersedes the dataset/scope sections of `PROJECT_GUIDE.md` for the new paper.
> Target: *Computers and Electronics in Agriculture*, Special Issue "Foundation Models in Agriculture".
> Hard internal deadline: **August 2026** (~10 weeks). Official SI deadline 30 Sep 2026.

---

## 0. One-paragraph thesis

Frontier vision-language models (e.g. SAGE's agentic diagnoser) reach high in-the-wild disease
accuracy but cost **$0.21–0.42 per image** and cannot run on a phone or Pi. We **distill that
foundation-model symptom knowledge into a 1.3M-param INT8 EdgeNeXt-XX-Small student** that runs on a
$35 device at zero marginal cost. **PRIMARY HEADLINE: cross-crop generalization** — the tiny student,
anchored on LLM-authored symptom-descriptor text prototypes, diagnoses crops it NEVER saw during
training (zero-shot via the text prototypes), which is the capability a foundation-model distillation
uniquely enables and the open problem (Ghazal 2024) the field has not solved at edge scale.
SUPPORTING results: in-the-wild accuracy on trained crops, an **OOD/abstain gate** for honest field
behavior, and a real **phone+Pi** efficiency benchmark. SAGE is the cloud teacher's knowledge source;
we are the deployable edge student that does the one thing the cloud agent can't do cheaply.

> **SI fit (why this framing maximizes acceptance):** the Special Issue is "Foundation Models in
> Agriculture." Zero-shot transfer to unseen crops via text prototypes is a *foundation-model-native*
> capability (impossible without the CLIP/VLM teacher) — it answers the SI theme directly, unlike a
> plain "small CNN, good accuracy" paper. Generalization is the lead claim; edge + abstention support it.

> **DATASET POLICY (locked by user):** SAGE-ONLY. Do NOT use PlantVillage (lab-only, saturated, 2016)
> or PlantDoc/PlantWild as train OR eval. This removes the old "lab→field" axis on purpose; the
> generalization axis is now the **held-out-crop split inside SAGE** (train 7 crops → test 4 unseen).

---

## 1. How we use SAGE (official sources + why we don't take all of it)

**Official endpoints (confirmed):**
- Code + symptom registry KB: `https://github.com/tirtho149/SAGE` (KB lives here, plus
  `open_agentic/prompt.py`).
- Images (HuggingFace): `https://huggingface.co/datasets/tirtho149/SAGE` —
  **~1.01M rows / ~114 GB, MIT license**, stored as **Parquet shards**, ONE flat `train` split,
  columns `{image: bytes, crop, disease, filename}`. **NOT per-crop downloadable.**

**Why not all of it:** 114GB is off-strategy AND infeasible for our Kaggle 9hr/storage limits. Our
paper is a *focused, field-deployable* model for a useful crop set, not maximal coverage (see
`memory/plant-disease-repo-facts.md` data philosophy).

**The pull strategy (corrected — STREAM + FILTER, do NOT download 114GB):**
Because HF serves it as a `datasets` repo, we stream rows and keep only our 8 crops up to a per-class
cap. Only the *kept* images (~6–8 GB) ever materialize; the full blob is never stored.
```python
from datasets import load_dataset
ds = load_dataset("tirtho149/SAGE", streaming=True, split="train")  # no full download
TRAIN = {"Tomato","Soybean","Apple","Corn","Grape","Potato","Rice"}      # 7 trained
HELDOUT = {"Coffee","Orange","Peach","Pumpkin"}                          # 4 zero-shot test
WANT = TRAIN | HELDOUT
# iterate → keep row["crop"] in WANT, cap per (crop,disease) → decode bytes → save jpg + manifest
```
**Run this ON Kaggle** (fast HF bandwidth; output becomes a Kaggle Dataset directly → feeds Phase B
natively; avoids the home connection entirely).

**License:** HF aggregate is **MIT** → fine to train/eval/cite a research model. Caveat: underlying
sub-sets keep their own terms (PlantWild/CDDM were NC-ND in the paper), so don't claim the whole
corpus as ours to rehost; for any redistributed derivative, prefer the clearly-permissive crops.

**Symptom KB:** take SAGE's GitHub registry as the seed (it's the descriptor goldmine), images we
take selectively via the stream-filter above.

---

## 2. The curated subset (LOCKED — SAGE-only, 7 train + 4 held-out)

Chosen from the thick, well-populated wedges of the SAGE sunburst (839K). All from SAGE; NO
PlantVillage / PlantDoc / PlantWild.

**TRAIN / DISTILL crops (7) — broad grower base, foundational coverage:**
Tomato, Soybean, Apple, Corn(Maize), Grape, Potato, Rice.
- Example dense classes (from sunburst): Tomato {Leaf Mosaic Virus, Target Spot, Leaf Mold, Early
  Blight, Bacterial Leaf Spot, Late Blight, Septoria Leaf Blotch, Tomato Leaf Curl Virus};
  Soybean {Frogeye, Bacterial Blight, Brown Spot, Bacterial Pustule, +deficiencies}; Apple {Cedar
  Apple Rust, Black Rot, Alternaria Blotch, Apple Mosaic Virus, Apple Scab, ...}; Corn {Common Rust,
  Northern Leaf Blight, Maize Streak Virus, Gray Leaf Spot}; Grape {Black Measles, Black Rot, Leaf
  Blight}; Potato {Late Blight, Alternaria, Bacterial Leaf Spot}; Rice {Bacterial Leaf Spot, Blast,
  Tungro, Brown Spot}.
- **Tomato = showcase crop** (highest inter-class similarity → where descriptor anchoring should
  beat a plain student; mirrors SAGE's own Tomato +KB gain).

**HELD-OUT ZERO-SHOT crops (4) — never trained, the PRIMARY generalization test:**
Coffee, Orange (Huanglongbing), Peach, Pumpkin. Distinctive, well-populated classes. The student
diagnoses these using ONLY the LLM-descriptor text prototypes (no training images) → the headline
foundation-model capability. Report zero-shot accuracy here as the main result.

**OOD / ABSTAIN probes:** the SAGE "+315 more" long-tail crops + any class with <50 images →
the gate must ABSTAIN on these (honest "I don't know"). No external weed/insect sets needed —
the long tail of SAGE itself is the OOD source.

**Per-crop budget:** cap ~**1,500 images/class**; drop classes <50 imgs to the OOD set.
Target **~80–110 classes, ~6–8 GB images, ~3–4 GB embedding cache** → fits Kaggle 9hr with margin.

**Within-crop split (trained crops):** random 80/10/10 train/val/test per class. SAGE mixes
controlled + in-the-wild images per class; where the manifest exposes source we additionally hold the
in-the-wild-looking images as a "harder conditions" test slice (recovers a difficulty axis without
external datasets).

**Symptom descriptors:** SAGE GitHub registry for ALL 11 crops (incl. the 4 held-out — descriptors
are how zero-shot works) as the seed, extended with Claude using SAGE's **source-grounded schema**:
every field = `{value, source_url, verbatim_quote}`, LLM forbidden from own-knowledge, expert
spot-audit a sample. Kills the hallucination critique and makes the headline novelty auditable.

---

## 3. Compute & Kaggle execution (unchanged constraints)

- Hardware: RTX 4060 (friend, first) + 3 Kaggle accounts (~90 GPU-hr/wk) + Colab.
- Kaggle rules: **Save Version → Run All (commit)** for background exec (survives tab close, 9hr cap,
  no 40-min idle death). Checkpoint+resume so every run fits <9hr.
- **The key trick:** compute frozen teacher embeddings ONCE → publish as a shared **Kaggle Dataset**
  (all 3 accounts read it). Student training then needs only cached vectors + small images → ~3–5
  hr/run. The embedding cache (~few GB) is what must fit, NOT the 114GB.

---

## 3b. Architecture (LOCKED) — teacher, student, and the zero-shot mechanism

**Critical design fact:** zero-shot to UNSEEN crops requires an **image–text aligned** space (image
can be compared to a descriptor prototype). CLIP-family teachers (CLIP/OpenCLIP/SigLIP2) have this;
**DINOv2 is vision-only and CANNOT do zero-shot** — it is an AUXILIARY teacher for trained-crop visual
robustness ONLY. So the headline rests on the CLIP-family alignment; DINOv2 is a supporting actor.
Be explicit about this in the paper (preempts the obvious reviewer question).

**Teachers (all FROZEN):**
- **Text-aligned (zero-shot engine) — BENCHMARK 3 in Phase B, pick winner:** CLIP ViT-B/16 (512-d,
  smallest cache, baseline) · OpenCLIP ViT-L/14 LAION-2B (768-d, strong, zero-risk) · **SigLIP2**
  (2025, SOTA open zero-shot — VERIFY weights load on Kaggle/timm in Phase A). The comparison
  ("which FM distills best into an edge student for zero-shot?") is itself a paper contribution.
- **Auxiliary (trained crops only):** DINOv2-small — vision-only structure/robustness signal.

**Student: EdgeNeXt-XX-Small (~1.3M params), TWO heads:**
1. **Text-projection head** → projects student image features into the winning teacher's text-aligned
   space. THIS IS THE ZERO-SHOT ENGINE: descriptors → text prototypes → cosine NN classifies any crop
   (seen or unseen). A no-descriptor student has no prototypes → 0% zero-shot → novelty is provably
   the enabling mechanism (the load-bearing claim).
2. **DINOv2-alignment head** (auxiliary) → regularizes visual structure; improves trained-crop +
   "harder conditions" accuracy. Can be dropped at inference.
- **Fallback if 1.3M underfits ~110 classes:** MobileNetV4-Conv-S (~2–3M) or TinyViT-5M. Decide
  empirically in Phase C — do NOT pre-commit to a struggling model.

**Story:** text-aligned FM → generalization; DINOv2 → robustness; distillation → a 1.3M INT8 model
that has both on a $35 device. Clean 3-way ablation: text-teacher-only vs +DINOv2 vs neither.

---

## 4. Reuse vs discard (from existing repo)

REUSE: training harness + ONNX export (`models/best_model.onnx` path/flow), FastAPI backend +
Next.js frontend (demo only), `scripts/consolidate_all_data.py` (real-image consolidation into
`dataset_cleaned/` — PARAMETERIZE its hardcoded `C:\Projects\Plant project\data\...` paths),
`scripts/download_kaggle_datasets.py` (pattern reference only — actual pull is HF stream-filter).

DISCARD as the paper's contribution: `research_paper.tex` (old IEEE 10-class tomato ensemble),
`scripts/build_multicrop_dataset.py` (synthetic 20K/crop augmentation — reviewers discount synthetic
data; we use REAL SAGE images + held-out-crop generalization instead), the `.pth` ensemble weights
(ResNet50/EffB4/ViT) as a contribution — keep only as a "heavy baseline" comparison point if useful.
ALSO NOT USED (user policy): PlantVillage, PlantDoc, PlantWild — SAGE-only.

---

## 5. Next plan (10-week, ordered)

### Phase A — Data foundation (Week 1–2)
- A1. Write `scripts/build_sage_subset.py` (run as a Kaggle notebook): `load_dataset(
  "tirtho149/SAGE", streaming=True, split="train")`, filter `crop ∈ TRAIN∪HELDOUT` (11 crops), cap
  per (crop,disease) ~1,500, decode `image` bytes → `dataset_cleaned/<Crop>___<Disease>/<filename>`,
  dedupe by content hash, drop classes <50 imgs (→ OOD list). Emit manifest CSV (path, crop, disease,
  filename, split_role ∈ {train_crop, heldout_crop}). Publish as a Kaggle Dataset. **Never download
  the full 114GB — stream + filter only.**
- A2. Clone `github.com/tirtho149/SAGE`, pull the symptom registry KB for ALL 11 crops (held-out
  included — descriptors are how zero-shot works) as the seed; write `scripts/build_descriptors.py`
  (Claude + source-grounded `{value,source_url,verbatim_quote}` schema, LLM forbidden from
  own-knowledge) → `descriptors/<crop>.json`. Expert spot-audit a sample.
- A3. Build the splits from the manifest ALONE (no external sets): trained-crop 80/10/10 per class;
  held-out crops reserved entirely for zero-shot eval; the SAGE long-tail "+315 more" + dropped
  <50-img classes form the OOD/abstain set. Verify ZERO image leakage between roles (hash check).
- A4. Friend smoke-tests the whole data build + a 1-epoch student run on the RTX 4060.

### Phase B — Teacher embedding cache (Week 2–3)
- B1. Kaggle notebook: embed all subset images ONCE with the 3 frozen text-aligned teachers
  (CLIP ViT-B/16, OpenCLIP ViT-L/14, SigLIP2) + DINOv2-small (aux). Compute the matching text-prototype
  embeddings from the descriptors for EACH text teacher. Checkpoint per shard (<9hr); teachers are
  frozen so this is ~3× caching, NOT 3× training.
- B2. Publish embeddings + per-teacher prototypes as a shared Kaggle Dataset. Verify it loads from a
  second account. This single asset feeds every student run.

### Phase C — Distillation student (Week 3–6)
- C0. **Teacher bake-off:** train the lightweight student head against each cached text teacher
  (CLIP-B/16 / OpenCLIP-L/14 / SigLIP2), measure zero-shot held-out accuracy, pick the winner as the
  primary teacher. Report the 3-way comparison as an ablation/contribution. (DINOv2 stays as aux.)
- C1. EdgeNeXt-XX-Small student (~1.3M params), two heads (§3b). 3-stage curriculum:
  (i) feature alignment to dual-teacher embeddings (winning text teacher + DINOv2 aux),
  (ii) text-prototype anchoring on LLM descriptors (this is what enables zero-shot),
  (iii) fine-tune on trained crops. If 1.3M underfits → fallback MobileNetV4-S / TinyViT-5M.
- C2. **PRIMARY EXPERIMENT — zero-shot held-out crops:** classify the 4 unseen crops by nearest
  LLM-descriptor text prototype (no training images for them). Report this as the headline result;
  compare against (a) the SAGE cloud agent's accuracy and (b) a no-descriptor student (which CANNOT
  do zero-shot at all → the descriptor anchoring is the enabling mechanism).
- C3. OOD/abstain gate = distance to nearest descriptor prototype (free). Calibrate threshold on a
  trained-crop val split; evaluate abstention on the SAGE long-tail OOD set + report
  accuracy-vs-coverage (risk-coverage) curve.
- C4. Ablations: no-descriptor vs descriptor (does it enable/boost zero-shot + mirror SAGE's +16.2pp
  on trained crops at edge scale?); single- vs dual-teacher (CLIP-only vs CLIP+DINOv2);
  naive-prompt vs source-grounded LLM-descriptor.

### Phase D — Edge optimization + benchmark (Week 6–8)
- D1. INT8 quantization (QAT or PTQ). Export ONNX → TFLite.
- D2. Real-device benchmark on BOTH Android (TFLite) + Raspberry Pi: params, GFLOPs, INT8 size,
  **p50/p95 latency, RAM, energy/inference**. Frame tiny-model efficiency AS a contribution
  (Gao YOLO-GPP precedent: 1.98M params / CPU-only FPS accepted as a result).

### Phase E — Writing (Week 8–10)
- E1. New `paper.tex` (discard IEEE draft). Reuse SAGE Table 1 as the datasets-comparison table.
  Lead the results section with the zero-shot held-out-crop table (the headline).
- E2. Rigor checklist: sign test / Wilcoxon for student-vs-baseline; Grad-CAM panels incl. under
  corruption (DCDD-RL blur/illumination/occlusion — cheap robustness curve, reconsider un-cutting);
  full metric set (P/R/F1, per-crop accuracy, risk-coverage curve, on-device latency).
- E3. Positioning: SAGE = accurate-but-$0.30/img cloud agent that needs reference images per crop;
  us = deployable INT8 edge distillation that generalizes to unseen crops via text prototypes at
  $0/img. Cite VLM-for-ag canon (AgroGPT, Agri-LLaVA, AgReason, ChatLeafDisease, PDD-Agent,
  PepperDet +9%). Frame zero-shot cross-crop transfer as the SI's "foundation model" contribution.

---

## 6. August cut list (unchanged)

IN: LLM-descriptor distillation, CLIP+DINOv2 dual teacher, OOD abstain gate, INT8, phone+Pi benchmark.
DEFERRED (mid-2027 extension): severity estimation (SAGE/DSI), full robustness corruption benchmark
(adopt the cheap OpenCV version if time permits), fruit-organ extension, few-shot cassava, deployed
farmer app (demo only).

See `memory/plant-disease-litreview.md` for the 12-paper synthesis and `memory/plant-disease-cea-paper.md`
+ `memory/plant-disease-repo-facts.md` for positioning and locked design.
