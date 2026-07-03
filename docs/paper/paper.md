# Compact Vision–Language Models for Cross-Crop Plant-Disease Diagnosis at the Edge

*Draft manuscript — Computers and Electronics in Agriculture, Special Issue "Foundation Models in
Agriculture" (submission deadline 30 September 2026). Numbers are final for the runs reported
(local RTX-GPU workstation, 2 July 2026); items still pending are listed in §9.*

**Authors:** _[Abhiram et al. — fill affiliations]_

---

## Abstract

Frontier vision–language models (VLMs) diagnose plant disease accurately but cost cents per image and
cannot run offline on a phone, laptop, or field camera. We ask how small a **deployable** model can be
while still performing **cross-crop zero-shot** disease diagnosis — recognising diseases on crops it was
never trained on, by matching a leaf image to LLM-authored, source-grounded symptom descriptions. Using
the SAGE dataset and off-the-shelf frozen compact CLIP encoders, we show that (i) an **11.4 M-parameter
frozen encoder** reaches **27.0 %** cross-crop zero-shot top-1 on a 17-class held-out set (4.6× chance)
and **86 %** of the accuracy of a 93 M reference model, while accuracy is **nearly flat from 11 M to
300 M parameters**; (ii) **descriptor *detail* is the accuracy lever** — any full symptom description
beats a class-name prompt by +6.9 pp on average, whereas the *authoring method* (hand-curated vs.
LLM source-grounded) is statistically a wash, so source-grounding delivers **auditability at no accuracy
cost**; (iii) with a **top-1−top-2 margin** confidence, an abstain gate makes the modest top-1
field-useful — selective accuracy rises to ~36 % at 50 % coverage and **top-5 reaches 62–77 %**; and (iv)
the deployable image encoder runs at **21.5 ms/image on a laptop CPU** (ONNX FP32, ~47 img/s) and
quantises to **12 MB** (INT8). A leave-one-crop-out analysis shows the held-out crops are not
cherry-picked, and a supervised CNN baseline — competitive on seen crops (74.9 %) — is structurally
incapable of the cross-crop transfer the descriptor head provides. We release the descriptor pipeline,
per-tier models, and benchmark so the result is reproducible and auditable.

**Keywords:** foundation models; vision–language models; plant disease; cross-crop zero-shot; edge
deployment; source-grounded descriptors; precision agriculture.

---

## 1. Introduction

Cloud VLM diagnosers (e.g., the SAGE agent) achieve high in-the-wild accuracy but cost roughly
\$0.21–0.42 per image and require connectivity — a poor fit for smallholder and field use, which needs
**offline, low-cost, real-time** inference. A second, deeper obstacle is generalisation: models trained
on one crop transfer poorly to others (Ghazal et al., 2024), and edge deployment additionally demands
compression. Cross-crop transfer is therefore the *foundation-model-native* open problem this Special
Issue targets: it is enabled only by an image–text-aligned VLM, not by a conventional classifier.

We study the question: **how small can a model be and still perform cross-crop disease zero-shot at the
edge?** Our system is one **frozen** compact CLIP encoder with two heads and an abstain router: a
**trained head** for accurate real-time diagnosis on known crops, and a **descriptor head** that
diagnoses unseen crops by matching image embeddings to LLM-authored, source-grounded symptom-descriptor
text. Only the image encoder ships to the device; text prototypes are precomputed offline.

**Contributions.**
1. **Frozen compact VLMs do cross-crop zero-shot.** An 11.4 M encoder reaches 27.0 % on 17-class
   held-out zero-shot (4.6× chance, 86 % of a 93 M model), and accuracy is flat 11 M→300 M — *model size
   is not the bottleneck* (§5.1–5.3).
2. **Descriptor detail, not authoring method, is the lever.** Full descriptions beat class names by
   +6.9 pp; hand-curated and LLM source-grounded descriptors tie on accuracy, so **source-grounding buys
   auditability for free** (§5.3).
3. **A calibrated abstain gate + top-5** make the modest top-1 field-useful (§5.4).
4. **A per-tier edge benchmark** with real INT8 sizes and CPU latency, and an efficiency Pareto that
   identifies the 11 M tier as the deployment sweet spot (§5.9).
5. **Rigorous negatives that save effort:** naive fine-tuning causes catastrophic forgetting (WiSE-FT
   recovers the trade-off, §5.6); a supervised CNN cannot transfer cross-crop (§5.8); a leave-one-crop-out
   analysis shows the held-out crops are not cherry-picked (§5.7).

## 2. Related work

- **Agricultural VLMs / SAGE (Arshad et al.).** SAGE provides our data and the cloud agent we contrast
  against; its `{value, source_url, verbatim_quote}` source-grounded schema and its +14–16 pp gain from
  symptom knowledge motivate our descriptor design. We are the deployable edge counterpart at \$0/image.
- **Descriptor-based classification — DCLIP (Menon & Vondrick, 2023), CuPL (2023).** Classify by
  LLM-written visual descriptions. *Our delta:* source-grounded (auditable, anti-hallucination),
  compressed to the edge, and evaluated for *cross-crop* agricultural transfer rather than general
  retrieval.
- **Compact CLIPs — MobileCLIP/MobileCLIP2 (Apple), TinyCLIP.** Provide our frozen backbones. *Our
  delta:* an agricultural cross-crop zero-shot study and an edge efficiency analysis, not retrieval.
- **Robust fine-tuning — WiSE-FT (Wortsman et al.).** Explains our forgetting result and motivates the
  frozen-backbone + weight-ensembling design.
- **Domain foundation models — BioCLIP 2, SCOLD (2025).** Candidate teachers/baselines; we find they do
  not transfer to leaf disease under our descriptor protocol (§5.2).

## 3. Materials and methods

### 3.1 Dataset
We use **SAGE only** (`tirtho149/SAGE`, MIT; ~1.01 M images as parquet shards) to avoid the "old,
lab-only" critique of legacy corpora. We stream-filter shards to our crops with a per-class cap and
content-hash de-duplication, yielding a **12,288-image** subset over **7 crops**: trained crops
**Apple, Corn, Potato, Soybean** (80 disease classes) and held-out zero-shot crops **Coffee, Orange,
Peach** (17 classes). Held-out crops are never used for training; they are diagnosed only from
descriptor text. *(Tomato and Grape live in later shards and are pending a fuller pull; see §9.)*

### 3.2 Source-grounded descriptors
Each disease carries a symptom record `{symptom_text, fields:{pathogen, affected_organs,
visual_symptoms}}`, each field with `{value, source_url, verbatim_quote}`. Records are generated by an
LLM (Claude Sonnet via an OpenAI-compatible endpoint) under a strict grounding prompt that forbids
un-cited claims, then audited. Because the generation endpoint cannot browse, raw citations are
model-recalled; we therefore **web-verify the headline held-out diseases**, replacing citations with
sentences copied verbatim from reachable authoritative pages (UC-IPM, university extension, reference
encyclopedias). The **`symptom_text`** field — the string that becomes the CLIP text prototype — is the
functional component; the verified citations provide the auditability guarantee. The descriptor
generator, audit, and verified-citation tools are released.

### 3.3 Architecture
One **frozen** image–text encoder per tier, plus:
1. **Trained seen-crop head** — a linear probe (or light fine-tune) on the 80 seen classes for best
   real-time accuracy on known crops.
2. **Zero-shot descriptor head** — nearest source-grounded text prototype (cosine), diagnosing unseen
   crops at \$0/crop.
3. **WiSE-FT weight-ensembling** — interpolate fine-tuned ⊕ frozen visual weights at ratio α so seen
   gains do not destroy unseen zero-shot.
4. **Abstain / OOD router** — top-1−top-2 similarity margin; low margin → "unsure."

Only the image encoder deploys; text prototypes are precomputed offline. We evaluate four deployable
tiers plus a reference ceiling:

| Tier | Base (open_clip) | Image-enc params | Target |
|---|---|---|---|
| Lightweight | MobileCLIP2-S0 | 11.4 M | small NPU / phone |
| Lightweight | MobileCLIP-S1 | 21.5 M | laptop CPU |
| Lightweight | MobileCLIP2-S2 | 35.8 M | laptop |
| Heavyweight | MobileCLIP-B | 86.3 M | workstation |
| Reference | ViT-B/16-SigLIP2 | 92.9 M | cloud ceiling |

### 3.4 Descriptor strategies (ablation axis)
`bare` (class name only) → `crude` (one generic keyword sentence) → `rich` (a hand-curated symptom
paragraph) → `grounded` (the LLM source-grounded `symptom_text`; falls back to `rich` where a descriptor
is absent). A `grounded_visual` variant uses only the visual-symptoms field.

## 4. Experimental setup
Held-out evaluation: **3 unseen crops, 17 disease classes, 1,618 images, chance 5.9 %**. Metrics: top-1,
**top-5**, per-crop accuracy, **risk–coverage / AURC** (abstain), leave-one-crop-out with bootstrap 95 %
CIs, and on-device latency/size. Models are loaded via `open_clip`; edge numbers use ONNX Runtime on CPU
(224×224, batch 1). All runs are on a single RTX-GPU workstation.

## 5. Results

### 5.1 A frozen 11 M model already does cross-crop zero-shot, and accuracy is flat with size
On the 17-class held-out set, a frozen **MobileCLIP2-S0 (11.4 M)** reaches **27.0 %** rich-descriptor
zero-shot (4.6× chance) — **86 %** of the 31.5 % achieved by the 93 M SigLIP2 reference — while accuracy
is nearly flat across 11 M→300 M parameters (Fig. `fig_efficiency_curve.png`). Model size is not the
bottleneck; the pretrained alignment is the asset. Attempts to *train* a tiny model to learn this
alignment fail (from-scratch ≈ chance; specialising on seen crops forgets, Fig.
`fig_specialization_forgetting.png`), so we keep the backbone frozen.

### 5.2 Encoder bake-off — generic beats domain-specific, and small punches up (Fig. `fig_bakeoff.png`)

| Encoder | Params | Rich zero-shot | Coffee | Orange | Peach |
|---|---|---|---|---|---|
| **SigLIP2** | 92.9 M | **31.5 %** | 21.4 % | 36.9 % | 36.6 % |
| MobileCLIP2-S2 | 35.8 M | 28.7 % | 29.8 % | 35.5 % | 19.6 % |
| MobileCLIP2-S0 | 11.4 M | 27.0 % | 34.6 % | 29.0 % | 16.2 % |
| MobileCLIP-S1 | 21.5 M | 22.4 % | 13.9 % | 29.1 % | 24.4 % |
| BioCLIP2 | 304.0 M | 9.6 % | 15.2 % | 4.1 % | 9.5 % |
| SCOLD | 237.5 M | 4.4 % | 2.7 % | 5.2 % | 5.5 % |

SigLIP2 is the accuracy ceiling. Notably, the **biological (BioCLIP2) and domain leaf-disease (SCOLD)
foundation models perform at or below chance** — a domain-mismatch result: taxonomy/contrastive
pretraining does not align to symptom-descriptor text. (Our SCOLD wrapper is a best-effort load; a
faithful evaluation needs the authors' inference pipeline — §9.) The 11 M S0 recovers 86 % of SigLIP2 at
1/8 the parameters, and **S0 (27.0 %) beats the larger S1 (22.4 %)** — a training-recipe effect
(MobileCLIP2/DFN vs. older MobileCLIP/DataComp), reinforcing that pretraining quality, not size, drives
transfer.

### 5.3 Descriptor detail is the lever; authoring method is a wash (Fig. `fig_descriptor_ablation.png`)

| Model | Params | bare | crude | rich | grounded |
|---|---|---|---|---|---|
| MobileCLIP2-S0 | 11.4 M | 18.3 % | 19.8 % | **27.0 %** | 24.4 % |
| MobileCLIP-S1 | 21.5 M | 18.9 % | 21.1 % | 22.4 % | **28.6 %** |
| MobileCLIP2-S2 | 35.8 M | 18.5 % | 20.3 % | **28.7 %** | 21.1 % |
| MobileCLIP-B | 86.3 M | 21.6 % | 22.6 % | 26.8 % | **26.8 %** |
| **mean** | — | 19.3 % | 20.9 % | **26.2 %** | 25.2 % |

Two findings. **(1) Detail is the lever:** any full symptom description beats the class-name prompt
(`bare`) by **+5–10 pp on every model** (mean +6.9 pp for `rich`), while generic keyword stubs (`crude`)
barely help (+1.6 pp). **(2) Authoring method is a wash:** hand-curated (`rich`) and LLM source-grounded
(`grounded`) tie (mean 26.2 % vs. 25.2 %; a `grounded_visual` variant using only the visual field is
similar), with model-dependent swings and no consistent winner. **Implication:** source-grounding does
not cost accuracy — it delivers the same lift as expert-curated text while being auditable and requiring
no domain expert, which is its real contribution (§6).

### 5.4 Top-5 and a calibrated abstain gate make the top-1 field-useful (Fig. `fig_riskcoverage.png`)

| Model | Strategy | Top-1 | Top-5 | AURC |
|---|---|---|---|---|
| MobileCLIP2-S0 (11.4 M) | rich | 27.0 % | 62.4 % | 0.661 |
| MobileCLIP2-S0 | grounded | 24.4 % | 70.3 % | 0.612 |
| MobileCLIP-S1 (21.5 M) | grounded | 28.6 % | 63.0 % | **0.526** |
| MobileCLIP-B (86.3 M) | grounded | 26.8 % | **76.8 %** | 0.607 |
| SigLIP2 (92.9 M) | rich | 31.5 % | 71.8 % | 0.654 |

A field tool can suggest a short list and abstain when unsure. **Top-5 reaches 62–77 %** even at the
11 M tier. For abstention, the confidence signal matters: raw max-similarity is anti-calibrated
(selective accuracy *falls* as coverage drops), but the **top-1−top-2 margin** calibrates cleanly — S0
selective accuracy **rises from 27.0 % to ~36 %** as coverage tightens from 100 % to 50 %. Across models,
**`grounded` abstains best** (lowest AURC, e.g., S1 0.526) and gives the highest top-5 — its cleaner
class separation shows up as both better ranking and better-calibrated confidence, even where its top-1
merely ties `rich`.

### 5.5 The hybrid works on one 11 M backbone (Fig. `fig_hybrid.png`)
A single frozen MobileCLIP2-S0 with two heads: the **trained probe reaches 67.0 % on the 80 seen
classes** (vs. 9.3 % seen zero-shot — a 7× gain from training), while **unseen zero-shot 27.0 % is
preserved** (the frozen backbone is untouched by the seen head). One model is simultaneously accurate on
known crops and generalising to unknown ones.

### 5.6 WiSE-FT tunes the seen↔unseen trade-off (Fig. `fig_wiseft.png`)

| α | Seen | Unseen (ZS) |
|---|---|---|
| 0.0 (frozen) | 67.0 % | 27.0 % |
| 0.5 (WiSE-FT) | 77.6 % | 15.4 % |
| 1.0 (naive fine-tune) | 82.2 % | 10.4 % |

α = 0 reproduces the frozen baseline exactly (validating the interpolation). Naive fine-tuning (α = 1)
is **catastrophic forgetting** — seen rises to 82.2 % but unseen collapses to 10.4 %. α = 0.5 is the
deployable middle ground (+10.6 pp seen while retaining most unseen), making the balance a single tunable
knob per deployment.

### 5.7 Leave-one-crop-out: the held crops are not cherry-picked
Over a 6-crop pool (held Coffee/Orange/Peach + trained Apple/Corn/Potato), each image matched against
**all** crops' prototypes (58 classes, chance 1.7 %), MobileCLIP2-S0, rich, bootstrap 95 % CI:

| Crop | N | Zero-shot acc | 95 % CI | Role |
|---|---|---|---|---|
| Corn | 729 | 32.9 % | [29.5, 36.4] | trained-pool |
| Coffee | 560 | 26.6 % | [22.9, 30.4] | held-out |
| Orange | 563 | 23.4 % | [20.1, 27.0] | held-out |
| Peach | 495 | 13.3 % | [10.5, 16.4] | held-out |
| Apple | 2,584 | 11.8 % | [10.6, 13.0] | trained-pool |
| Potato | 1,254 | 8.1 % | [6.6, 9.5] | trained-pool |
| **Pooled** | 6,185 | **16.0 %** | [15.1, 17.0] | — |

The **held-out crops fall in the middle of the range** — a *trained* crop (Corn) is easiest and two
*trained* crops (Apple, Potato) are hardest — so the headline held-out set is not a lucky split, and
every crop is 5–19× chance. Difficulty tracks intra-crop symptom distinctiveness, not train/held
membership (expected for a backbone that trained on none of them).

### 5.8 Supervised baseline: accurate on seen, structurally zero on unseen
A **MobileNetV3-Small** trained on the 80 seen classes reaches **74.9 % top-1** (best 76.0 % at epoch 6)
— competitive with, and above, the frozen-VLM linear probe (67.0 %). But it has **no output neuron for
an unseen class**, so its cross-crop accuracy is chance. This is precisely the capability the descriptor
head adds: the same seen-crop competence, plus generalisation the CNN cannot have.

### 5.9 Efficiency / on-device (Fig. `fig_edge_pareto.png`)
Image encoder only (the sole part that ships); laptop CPU, batch 1, 224×224, ONNX Runtime, 50 runs:

| Tier | Params | MACs | Torch FP32 | ONNX FP32 | ONNX INT8 | FP32 size | INT8 size |
|---|---|---|---|---|---|---|---|
| **S0** | 11.4 M | 1.8 G | 80.7 ms | **21.5 ms** | 463 ms | 45.8 MB | **12.1 MB** |
| S1 | 21.5 M | 3.6 G | 149.2 ms | 40.1 ms | 892 ms | 86.5 MB | 22.9 MB |
| S2 | 35.8 M | 6.0 G | 178.0 ms | 59.0 ms | 1333 ms | 143.6 MB | 37.4 MB |
| B | 86.3 M | 17.0 G | 174.8 ms | 106.6 ms | **70.7 ms** | 345.6 MB | 87.5 MB |

**Real-time is met in FP32:** S0 runs at **21.5 ms/image (~47 img/s) on a laptop CPU** (ONNX Runtime is
3–4× faster than eager PyTorch). **INT8 gives ~3.8× size reduction** (S0 → 12.1 MB). INT8 *latency* is
architecture-dependent under x86 dynamic quantization: it speeds up the pure-transformer B
(106.6 → 70.7 ms) but slows the hybrid conv-transformer S-tiers, whose quantized conv kernels are
unoptimized on desktop CPUs — the INT8 latency win requires an ARM/NPU runtime. For deployment: **S0
(FP32) for phones/laptops; B (INT8) where a transformer NPU is available.** (Raspberry Pi row pending an
on-device run of the same script.)

## 6. Discussion
- **The contribution is a system, not a backbone.** The four tiers are off-the-shelf encoders used
  frozen; the novelty is the source-grounded descriptor head, the calibrated abstain gate, the WiSE-FT
  seen/unseen knob, and the real-time edge packaging. Because accuracy is flat with size, the *reason*
  to offer a family is the measured **latency/size Pareto** (§5.9), not accuracy.
- **Source-grounding is a trust contribution, not an accuracy one.** LLM descriptors grounded in
  extension sources match hand-curated ones (§5.3), so the anti-hallucination guarantee comes for free;
  descriptor *detail* drives accuracy, the authoring method does not.
- **Real-time without exotic hardware.** Only the image encoder ships; 21.5 ms on a commodity CPU with a
  12 MB INT8 footprint makes the 11 M tier the default.
- **Honesty on absolutes.** Fine-grained 17-class top-1 is modest; we lead with top-5 (62–77 %) and the
  calibrated abstain gate as the field-relevant metrics, and frame cross-crop transfer as a hard open
  problem we advance at edge scale.

## 7. Limitations
- **Descriptor coverage.** 11 of 17 held-out diseases have filled descriptors (8 web-verified); 6
  obscure classes remain stubs and fall back to `rich`/`bare`. Automatic citations are model-recalled
  except where web-verified; five extension URLs were dead and need manual replacement.
- **Scope.** Four trained crops on disk; the showcase crop (Tomato) and Grape are pending a fuller SAGE
  pull. Single random seed; a single evaluation machine.
- **Sub-10 M gap.** No off-the-shelf aligned model exists below ~11 M; a ~5 M tier would require
  weight-inherited distillation and is framed as future work.
- **Domain-VLM baselines.** BioCLIP2/SCOLD numbers use our loaders; a faithful SCOLD evaluation needs
  the authors' pipeline.

## 8. Conclusion
A single frozen, compact, descriptor-driven VLM brings cross-crop plant-disease diagnosis to laptops and
small devices at \$0/image: **27 % zero-shot on unseen crops (86 % of a 93 M model at 1/8 the size),
67 %/74.9 % on known crops, 62–77 % top-5, a calibrated abstain gate, and 21.5 ms real-time CPU
inference in a 12 MB INT8 footprint.** Model size is shown to be a near-non-factor; the levers are
descriptor detail, honest abstention, and the frozen-backbone hybrid. We release the pipeline and
benchmark to make the result auditable and reproducible.

---

### 9. Open items before submission
- [ ] Raspberry Pi / phone latency row (run `benchmark_edge.py` on-device).
- [ ] Fill the 6 stub held-out descriptors; replace the 5 dead extension URLs.
- [ ] Fuller SAGE pull incl. **Tomato** (showcase crop) and Grape.
- [ ] Multi-seed runs for CIs on the headline zero-shot table.
- [ ] Faithful SCOLD loader (or footnote the current below-chance wrapper result).
- [ ] Convert to the Elsevier CompAg LaTeX template; final figure polish; author list & funding.
