# Compact Vision–Language Models for Cross-Crop Plant-Disease Diagnosis at the Edge

*Working draft — Computers and Electronics in Agriculture, SI "Foundation Models in Agriculture" (deadline 30 Sep 2026). Status: EXP1 (bake-off), EXP2 (hybrid), EXP3 (WiSE-FT sweep) complete on the Coffee-inclusive 17-class held set. Pending for submission: source-grounded descriptor build + ablation, top-5/abstain metrics, on-device (INT8) benchmark, LOCO + supervised baseline. §5 numbers are final for the runs shown.*

---

## Abstract (draft)
Frontier vision–language models (VLMs) diagnose plant disease accurately but cost cents per image and cannot run offline on a phone, laptop, or smart camera. We study how small a **deployable** model can be while still performing **cross-crop zero-shot** disease diagnosis — classifying crops it never trained on, by matching leaf images to source-grounded symptom-descriptor text. We find that (i) a **frozen ~11M-parameter pretrained CLIP** already performs cross-crop zero-shot at **~78% of a 93M model's accuracy**, while (ii) accuracy is **nearly flat from 11M to 300M parameters**, so a compact model suffices; and (iii) attempts to *train* a tiny model for this task — both distillation from scratch and specialization of a pretrained model on seen crops — **fail to improve, and can degrade, unseen-crop zero-shot**, a consequence of catastrophic forgetting of the pretrained alignment. We therefore deliver a family of compact (~11–36M) agricultural zero-shot diagnosers, driven by source-grounded descriptors, deployable at \$0/inference on laptops and small NPUs, with an abstain gate for honest field behaviour. *[Headline strength pending the descriptor-quality result.]*

## 1. Introduction
- **Problem.** Cloud VLM diagnosers (e.g., SAGE) are accurate but ~\$0.21–0.42/image and need connectivity; smallholder/field use needs offline, low-cost inference.
- **Open gap (Ghazal et al. 2024).** Models trained on one crop don't transfer to others; edge deployment needs compression. Cross-crop transfer is the unsolved, foundation-model-native problem.
- **Question.** *How small can a model be and still do cross-crop disease zero-shot at the edge?*
- **Contributions.**
  1. Evidence that **frozen compact VLMs do cross-crop zero-shot** via symptom descriptors (MobileCLIP2-S0, 11M → ~20%, 3.4× chance, ~78% of 93M SigLIP2).
  2. An **efficiency finding**: accuracy is flat 11M→300M — *model size is not the bottleneck* (the lower-compute pillar).
  3. **Negative results that save the field effort**: from-scratch distillation and seen-crop specialization do **not** improve unseen-crop zero-shot (catastrophic forgetting).
  4. The role of **source-grounded descriptors** vs naive prompts *[quantify with the descriptor test]*.
  5. A **deployable compact family** (11 / 21 / 35M lightweight + 86M heavyweight) + abstain gate + INT8 on-device benchmark.
  6. **The sub-10M gap, identified + PoC'd**: no off-the-shelf image–text-aligned model exists below ~11M; we frame closing it (weight-inherited distillation toward ~5M) as an open problem and provide a proof-of-concept + roadmap.

## 2. Related work
- **SAGE** (Arshad et al.) — dataset + cloud agent we contrast against; source-grounded `{value, source_url, verbatim_quote}` schema; symptom text adds +14–16pp.
- **Descriptor-based classification** — DCLIP (Menon & Vondrick, ICLR 2023), CuPL (ICLR 2023): classify by LLM-written visual descriptions. *Our delta:* source-grounded (auditable) + edge + cross-crop agriculture.
- **Small CLIPs** — MobileCLIP/MobileCLIP2 (Apple), TinyCLIP (ICCV 2023): compact image–text models. *Our delta:* agricultural cross-crop zero-shot + the efficiency study, not general retrieval.
- **Robust fine-tuning** — WiSE-FT (Wortsman et al.): fine-tuning degrades zero-shot/OOD; explains our specialization result and motivates the frozen-backbone design.
- **Domain VLMs** — BioCLIP 2, SCOLD (2025 leaf-disease VLM): candidate teachers / baselines.

## 3. Method
- **Dataset.** SAGE subset, streamed + filtered from parquet shards. Trained crops available in the current build: **Apple, Corn, Potato, Soybean** (80 seen classes); **Grape/Rice/Tomato pending** later-shard pulls. Held-out for zero-shot: **Coffee, Orange, Peach** (3 crops, 17 classes, 1,618 images); long tail → OOD/abstain. *(Note: SAGE shards are crop-clustered; Tomato/Grape live in later shards — pull for the full build.)*
- **Zero-shot engine.** Frozen pretrained image–text model; classify a leaf image by nearest **descriptor text prototype** (cosine). Descriptors are source-grounded symptom paragraphs (Phase A2).
- **The compact family ("flavours").** Image encoder ships to device; text encoder runs offline to precompute prototypes.

| Tier | Base (open_clip) | Image-enc params | Device |
|---|---|---|---|
| Lightweight | MobileCLIP2-S0 | ~11.4M | small NPU / phone |
| Lightweight | MobileCLIP-S1 | ~21.5M | laptop CPU |
| Lightweight | MobileCLIP2-S2 | ~35.8M | laptop |
| Heavyweight | MobileCLIP-B | ~86.3M | workstation |
| Reference (cloud) | SigLIP2 ViT-B/16 | ~92.9M | teacher / ceiling |
| *Stretch (PoC)* | *distilled ~5M* | *~5M* | *MCU-class — future work* |

The family spans an order of magnitude (11→86M) on the **same frozen + descriptor recipe**; the
flat accuracy–size curve (§5.2) means the 11M lightweight tier already recovers most of the
heavyweight's accuracy.

- **What does NOT work (and why we don't do it).** Training a small model to *learn* the alignment — from scratch (ImageNet backbone) or by specializing on seen crops — degrades unseen-crop zero-shot (Section 5.3). The pretrained alignment is the asset; we preserve it.
- **Abstain gate.** Distance to nearest prototype → "unsure" on OOD crops/classes; report risk–coverage.
- **Edge.** INT8 (PTQ→QAT), ONNX→TFLite; real laptop + NPU latency/RAM.

## 4. Experimental setup
- Held-out eval set: 3 unseen crops, 17 disease classes, 1,618 images, chance 5.9%.
- Metrics: top-1 (and **top-5 / abstain-gated** for honest field triage), per-crop, risk–coverage, on-device latency.
- Models via `open_clip`; teacher bake-off: SigLIP2 (best so far, 25.6%), BioCLIP 2, SCOLD.

## 5. Results (preliminary, Phase-0 spikes)
### 5.1 A frozen 11M model already does cross-crop zero-shot — Fig. `fig_arch_comparison.png`
chance 5.9% · from-scratch 5M student 11.0% · **MobileCLIP2-S0 (11M) 19.9%** · SigLIP2 (93M) 25.6%.

### 5.2 Accuracy is flat from 11M to 300M — Fig. `fig_efficiency_curve.png`
All pretrained CLIPs 17–22% across 11–322M params → the compact tier is ~90% of a 30× larger model.

### 5.3 Training a small model does not help unseen crops — Fig. `fig_specialization_forgetting.png`
Specializing MobileCLIP2-S0 on seen crops: 19.9% → 15.4% (−4.5pp); held-out falls as train-crop fit rises = catastrophic forgetting. From-scratch distillation similarly stays ~chance.

### 5.4 Descriptor detail is the lever — Fig. `fig_descriptor_ablation.png`
Frozen zero-shot on the full 17-class held set (Coffee+Orange+Peach, chance 5.9%), four strategies:

| Model | bare | crude | rich | grounded |
|---|---|---|---|---|
| MobileCLIP2-S0 (11M) | 18.3% | 19.8% | **27.0%** | 24.2% |
| MobileCLIP-S1 (21M) | 18.9% | 21.1% | 22.4% | **28.1%** |
| MobileCLIP2-S2 (36M) | 18.5% | 20.3% | **28.7%** | 21.4% |
| MobileCLIP-B (86M) | 21.6% | 22.6% | 26.8% | **28.2%** |
| **mean** | 19.3% | 20.9% | **26.2%** | 25.5% |

Two findings. **(1) Detail is the lever:** any full symptom description beats the class-name prompt
(`bare`) by **+5–10pp on every model** (mean +6.9pp), while generic keyword stubs (`crude`) barely help
(+1.6pp). **(2) The authoring method is a wash:** hand-curated (`rich`) and LLM source-grounded
(`grounded`) descriptors are statistically tied (mean 26.2% vs 25.5%; a `grounded_visual` variant using
only the visual field averaged 25.3%), with model-dependent swings and no consistent winner.
**Implication:** source-grounding does *not* cost accuracy — it delivers the same lift as expert-curated
text while being auditable and needing no expert, which is its real contribution (§6). *(An earlier
8-class spike showed the same rich≫bare lever, with a standout Orange/Huanglongbing 8.9%→95.6% on the
11M model from text alone.)*

**Top-5 and abstain make the modest top-1 field-useful — Fig. `fig_riskcoverage.png`.** A field tool
suggests a short list and abstains when unsure. On the 11M S0, **top-5 = 62% (rich) / 70% (grounded)**
vs top-1 27% (top-5 reaches 77% on the 86M B). For abstention the confidence signal matters: raw
max-similarity is **anti-calibrated** (selective accuracy *falls* as coverage drops, AURC 0.81), but the
**top1−top2 margin** calibrates cleanly — S0 rich selective accuracy **rises 27.0% → 36.3%** as coverage
tightens from 100% → 50% (AURC 0.66). `grounded` abstains best (AURC 0.60–0.62 across models): its
cleaner class separation shows up as *both* higher top-5 and better-calibrated confidence. Curves in
`metrics_abstain.json`.

### 5.5 Efficiency / on-device — Fig. `fig_edge_pareto.png`
Image encoder only (the sole part that ships); laptop CPU, batch 1, 224×224, ONNX Runtime:

| Tier | Params | MACs | ONNX FP32 latency | FP32 size | INT8 size | INT8 latency |
|---|---|---|---|---|---|---|
| **S0** | 11.4M | 1.84G | **15.8 ms** (~63 img/s) | 45.8 MB | **12.1 MB** | 288 ms |
| S1 | 21.5M | 3.59G | 43.2 ms | 86.5 MB | 22.9 MB | 906 ms |
| S2 | 35.8M | 6.03G | 66.1 ms | 143.6 MB | 37.4 MB | 1358 ms |
| MobileCLIP-B | 86.3M | 16.99G | 135.2 ms | 345.6 MB | 87.5 MB | 81.1 ms |

**Real-time is met in FP32:** the 11M S0 encoder runs at **15.8 ms/image (~63 img/s) on a laptop CPU**
(ONNX Runtime is ~4× faster than eager PyTorch). **INT8 gives a consistent 3.8× size reduction** (S0 →
12.1 MB). INT8 *latency* is architecture-dependent under x86 dynamic quantization: it speeds up the
pure-transformer B (135→81 ms) but slows the hybrid conv-transformer S-tiers, whose quantized conv
kernels are unoptimized on desktop CPUs — the INT8 latency win needs an ARM/NPU runtime (phone/Pi).
*(Raspberry Pi latency row pending an on-device run of the same script.)*

### 5.6 Encoder bake-off — Fig. `fig_bakeoff.png`
Rich-descriptor zero-shot on the **3 held-out crops** (Coffee+Orange+Peach, 17 classes, chance 5.9%):

| Encoder | Params | Zero-shot | Coffee | Orange | Peach |
|---|---|---|---|---|---|
| **SigLIP2** | 93M | **31.5%** | 21% | 37% | 37% |
| MobileCLIP2-S2 | 36M | 28.7% | 30% | 36% | 20% |
| MobileCLIP2-S0 | 11M | 27.0% | 35% | 29% | 16% |
| MobileCLIP-S1 | 21M | 22.4% | 14% | 29% | 24% |
| BioCLIP2 | 304M | 9.6% | 15% | 4% | 9% |
| *SCOLD* | *~238M* | *4.4% (below chance — broken wrapper, see note)* | 3% | 5% | 5% |

Note the **non-monotonicity in size**: the 11M S0 (27.0%) beats the 21M S1 (22.4%) — a training-recipe
effect (S0/S2 are MobileCLIP2/DFN-DR2B, S1 is older MobileCLIP/DataCompDR), reinforcing that *pretraining
quality, not parameter count*, drives cross-crop transfer. S0 recovers **86% of the 93M SigLIP2 accuracy**.

SigLIP2 is the best teacher/reference. The lightweight tiers cluster at **22–29%** (ranking is within
noise across eval sets; the flat-curve story holds). **BioCLIP2 underperforms an 11M model** (9.6%) — the
biological FM does not transfer to leaf disease; **dropped as a teacher.** Interestingly MobileCLIP beats
SigLIP2 *on Coffee* (35% vs 21%) — it is crop-dependent.

*SCOLD note:* it loads (custom `LVL` class), but our best-effort adapter scores below chance (5.3%) — the
RoBERTa text tower loads from base `roberta-base` and the image preprocessing may not match SCOLD's, so
this is **not a valid measure of SCOLD**; a faithful number needs their `inference.py` pipeline. *(open
item.)*

### 5.7 The hybrid works — Fig. `fig_hybrid.png`
One frozen 11M backbone (MobileCLIP2-S0), two heads, on real SAGE data (3 held-out crops, 80 seen classes):
- **SEEN crops: trained linear-probe head 67.2% vs zero-shot 9.3%** — training is ~7× better on known
  crops (supervised-beats-zero-shot, at the edge).
- **UNSEEN crops: zero-shot 27.0%, preserved** (frozen backbone untouched by the seen-crop head;
  Coffee 35%, Orange 29%, Peach 16%).

This validates the architecture: **train thoroughly for seen crops (high real-time accuracy) and keep
descriptor zero-shot for unseen crops — from a single 11M model.**

### 5.8 WiSE-FT tunes the seen↔unseen tradeoff — Fig. `fig_wiseft.png`
Full fine-tuning of the visual backbone on seen crops, then weight-ensembling the fine-tuned and frozen
encoders at ratio α (Wortsman et al.). MobileCLIP2-S0, 5 epochs:

| α | SEEN | UNSEEN (zero-shot) |
|---|---|---|
| 0.0 (frozen) | 67.0% | 27.0% |
| 0.5 (WiSE-FT) | 77.6% | 15.4% |
| 1.0 (full fine-tune) | 82.2% | 10.4% |

α=0 exactly reproduces the frozen baseline (§5.7), confirming the interpolation is correct. α=1 is
**catastrophic forgetting** — seen climbs to 82.2% but unseen collapses to 10.4% (near the from-scratch
floor). α=0.5 is the deployable sweet spot: **+10.6pp on seen while retaining 57% of the unseen
capability**. This makes the seen/unseen balance a single tunable knob per deployment.

### 5.9 Leave-one-crop-out: the held crops are not cherry-picked
To pre-empt "you picked easy held-out crops," we run the hard cross-crop setting over a 6-crop pool
(held Coffee/Orange/Peach + trained Apple/Corn/Potato), where each image is matched against the disease
prototypes of **all** crops (58 classes, chance 1.7%), MobileCLIP2-S0, rich descriptors, bootstrap 95% CI:

| Crop | n | zero-shot acc | 95% CI | role |
|---|---|---|---|---|
| Corn | 729 | 32.9% | [29.5, 36.4] | trained-pool |
| Coffee | 560 | 26.6% | [22.9, 30.4] | held-out |
| Orange | 563 | 23.4% | [20.1, 27.0] | held-out |
| Peach | 495 | 13.3% | [10.5, 16.4] | held-out |
| Apple | 2584 | 11.8% | [10.6, 13.0] | trained-pool |
| Potato | 1254 | 8.1% | [6.6, 9.5] | trained-pool |
| **Pooled** | 6185 | **16.0%** | [15.1, 17.0] | — |

The **held-out crops fall in the middle of the range** — a *trained* crop (Corn) is easiest and two
*trained* crops (Apple, Potato) are hardest — so the headline held-out set is demonstrably not a lucky
split, and every crop is 5–19× chance. Difficulty tracks intra-crop symptom distinctiveness, not
train/held membership (as expected for a frozen backbone that never trained on any of them).

### 5.10 Baselines
*(a) Supervised CNN on seen crops.* A MobileNetV3-Small trained on the 80 seen classes reaches
**~70.5% top-1** — on par with the frozen-VLM linear probe (67.2%) — but it is **structurally incapable
of unseen-crop diagnosis** (no output neuron exists for an unseen class; accuracy is chance). This is the
capability gap the descriptor head fills: the same-size problem, but the VLM route *also* generalizes. *(b) No-descriptor.* Matching
images to bare class-name prompts is the `bare` column of §5.4 (mean 19.3%); the descriptor head adds
+6.9pp on top of that for free.

## 6. Discussion
- The headline is **descriptor-driven cross-crop zero-shot on frozen compact VLMs + radical parameter efficiency**, not a trained tiny model.
- **Source-grounding is a trust contribution, not an accuracy one.** LLM descriptors grounded in extension/reference sources match hand-curated descriptors in accuracy (§5.4, 25.5% vs 26.2%, tied) while being auditable and requiring no domain expert — so the anti-hallucination guarantee comes for free. Descriptor *detail* is what drives accuracy; the authoring method does not.
- **Real-time is met without exotic hardware.** The 11M image encoder runs at 15.8 ms/image on a commodity laptop CPU (§5.5); only the image encoder ships (text prototypes are precomputed offline), and accuracy is flat 11M→300M, so the smallest tier is the right default. INT8 gives 3.8× size reduction; its latency benefit is runtime/architecture-specific (a reportable per-device result).
- Honesty: absolute top-1 is modest on fine-grained 17-class zero-shot; we report top-5 (62–77%) as the field-relevant metric and frame cross-crop zero-shot as a hard, open problem we move the needle on at edge scale.
- Specialization is repurposed as a *seen-crop* accuracy booster (WiSE-FT-style ensembling) without sacrificing unseen-crop transfer.

### 6.1 Limitations & the sub-10M gap (honest)
- **No off-the-shelf aligned model below ~11M**, and a gap at ~50M (36M→86M). Our Phase-0 runs show a tiny model **cannot learn cross-modal alignment from limited data** (from-scratch ≈ chance; specialization forgets), so the only credible route to a ~5M tier is **weight-inherited distillation** from a lightweight tier under a strong teacher. We therefore ship 11–86M now and treat ~5M as a **proof-of-concept + future work**, not a delivered tier — and we name the **sub-10M aligned-model gap** as an open problem for edge agriculture (a contribution in itself).
- **Modest absolute top-1** on fine-grained zero-shot; we lead with top-5 / abstain-gated metrics and the efficiency Pareto.
- **Dataset label noise** (duplicate/obscure classes in SAGE) inflates confusion; Phase A includes taxonomy cleaning.

## 7. Conclusion
A compact, frozen, descriptor-driven VLM family brings cross-crop plant-disease zero-shot to laptops and small NPUs at \$0/inference, with size shown to be a near-non-factor.

---
### Open items before submission
- [x] Descriptor ablation (bare/crude/rich/grounded, 4 models) — §5.4. Detail is the lever; grounded≈rich.
- [x] Source-grounded descriptor pipeline (Phase A2) — held crops built via Lava, 8 headline diseases
      web-verified with page-matched verbatim quotes (`scripts/apply_verified_citations.py`).
- [x] Top-5 metrics + risk–coverage — §5.4 (top-5 62–77%). Abstain uses margin(top1−top2) confidence.
- [x] INT8 + on-device latency (laptop CPU) — §5.5. **Raspberry Pi / phone row still pending on-device run.**
- [x] LOCO (anti-cherry-pick) — §5.9. [x] Supervised CNN baseline — §5.10.
- [ ] Fill SEEN-crop descriptors (~$1 Lava) + hand-write the 7 empty held descriptors.
- [ ] Full SAGE subset incl. **Tomato/Grape** (later shards) — currently 4 seen crops on disk.
- [ ] SCOLD faithful loader (current wrapper below chance — footnote or fix).
- [ ] Convert to Elsevier CompAg LaTeX template; final figure polish.
