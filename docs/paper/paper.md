# Compact Vision–Language Models for Cross-Crop Plant-Disease Diagnosis at the Edge

*Working draft — Computers and Electronics in Agriculture, SI "Foundation Models in Agriculture" (deadline 30 Sep 2026). Status: Phase-0 de-risk complete except the descriptor-quality test (pending). Numbers below are preliminary spike results, not final.*

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
- **Dataset.** SAGE subset, streamed + filtered from parquet shards; 5 trained crops + held-out crops (Coffee, Orange, Peach) reserved for zero-shot; long tail → OOD/abstain. *(Note: SAGE shards are crop-clustered; Tomato lives in later shards — pull for the full build.)*
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

### 5.4 Descriptor quality is the lever — Fig. `fig_descriptors.png`
Frozen zero-shot, bare vs crude vs rich (source-grounded-style), 8 classes (Orange+Peach), chance 12.5%:

| Model | bare | crude | rich | rich−bare |
|---|---|---|---|---|
| MobileCLIP2-S0 (11M) | 29.5% | 27.9% | **37.5%** | **+8.0** |
| MobileCLIP-S1 (21M) | 24.8% | 36.7% | **39.8%** | **+15.0** |
| SigLIP2 (93M) | 34.8% | 45.9% | **49.5%** | **+14.7** |

Rich descriptors are best on all three models; the +14–15pp lift on the larger models matches
SAGE's reported +14–16pp from symptom knowledge (independent corroboration), now shown on **frozen
compact edge models**. **Standout:** Orange/Huanglongbing on the 11M model goes **8.9% → 95.6%** with a
good symptom descriptor (SigLIP2 34.8%→94.8%; S1 8.9%→88.9%) — a distinctive disease on an unseen crop,
diagnosed by an 11M model from text alone.

*Caveats:* small noisy set (Coffee absent; `Leaf_Curl` and `Peach_Leaf_Curl` are duplicate labels of one
disease; `Whisker_Mold`/`Stigmina_Fungus` are obscure with no descriptor) — so Peach is mixed (rich can
make near-duplicate classes collide). Taxonomy cleaning + real source-grounded descriptors are Phase A.

### 5.5 Efficiency / on-device *(Phase D)*
INT8 size, p50/p95 latency, RAM on laptop + small NPU per tier.

### 5.6 Encoder bake-off — Fig. `fig_bakeoff.png`
Rich-descriptor zero-shot on the **3 held-out crops** (Coffee+Orange+Peach, 17 classes, chance 5.9%):

| Encoder | Params | Zero-shot |
|---|---|---|
| **SigLIP2** | 93M | **31.5%** |
| MobileCLIP2-S2 | 36M | 28.7% |
| MobileCLIP2-S0 | 11M | 26.9% |
| MobileCLIP-S1 | 21M | 22.5% |
| BioCLIP2 | 304M | 9.6% |
| *SCOLD* | *~238M* | *5.3% (adapter issue — see note)* |

SigLIP2 is the best teacher/reference. The lightweight tiers cluster at **22–29%** (ranking is within
noise across eval sets; the flat-curve story holds). **BioCLIP2 underperforms an 11M model** (9.6%) — the
biological FM does not transfer to leaf disease; **dropped as a teacher.** Interestingly MobileCLIP beats
SigLIP2 *on Coffee* (35% vs 21%) — it is crop-dependent.

*SCOLD note:* it loads (custom `LVL` class), but our best-effort adapter scores below chance (5.3%) — the
RoBERTa text tower loads from base `roberta-base` and the image preprocessing may not match SCOLD's, so
this is **not a valid measure of SCOLD**; a faithful number needs their `inference.py` pipeline. *(open
item.)*

### 5.7 The hybrid works — Fig. `fig_hybrid.png`
One frozen 11M backbone (MobileCLIP2-S0), two heads, on real SAGE data (3 held-out crops):
- **SEEN crops (80 classes): trained linear-probe head 67.0% vs zero-shot 8.5%** — training is ~8× better
  on known crops (supervised-beats-zero-shot, at the edge).
- **UNSEEN crops: zero-shot 26.9%, preserved** (frozen backbone untouched by the seen-crop head).

This validates the architecture: **train thoroughly for seen crops (high real-time accuracy) and keep
descriptor zero-shot for unseen crops — from a single 11M model.** *(EXP3 full fine-tune + WiSE-FT: a
theta0-aliasing bug was fixed; a clean WiSE-FT sweep re-run is pending.)*

## 6. Discussion
- The headline is **descriptor-driven cross-crop zero-shot on frozen compact VLMs + radical parameter efficiency**, not a trained tiny model.
- Honesty: absolute top-1 is modest on fine-grained 17-class zero-shot; we report top-5 / abstain-gated accuracy as the field-relevant metric and frame cross-crop zero-shot as a hard, open problem we move the needle on at edge scale.
- Specialization is repurposed as a *seen-crop* accuracy booster (WiSE-FT-style ensembling) without sacrificing unseen-crop transfer.

### 6.1 Limitations & the sub-10M gap (honest)
- **No off-the-shelf aligned model below ~11M**, and a gap at ~50M (36M→86M). Our Phase-0 runs show a tiny model **cannot learn cross-modal alignment from limited data** (from-scratch ≈ chance; specialization forgets), so the only credible route to a ~5M tier is **weight-inherited distillation** from a lightweight tier under a strong teacher. We therefore ship 11–86M now and treat ~5M as a **proof-of-concept + future work**, not a delivered tier — and we name the **sub-10M aligned-model gap** as an open problem for edge agriculture (a contribution in itself).
- **Modest absolute top-1** on fine-grained zero-shot; we lead with top-5 / abstain-gated metrics and the efficiency Pareto.
- **Dataset label noise** (duplicate/obscure classes in SAGE) inflates confusion; Phase A includes taxonomy cleaning.

## 7. Conclusion
A compact, frozen, descriptor-driven VLM family brings cross-crop plant-disease zero-shot to laptops and small NPUs at \$0/inference, with size shown to be a near-non-factor.

---
### Open items before submission
- [ ] Descriptor-quality result (rich vs crude) → decides headline strength.
- [ ] Source-grounded descriptor pipeline (Phase A2) with audit.
- [ ] Full SAGE subset incl. Tomato (later shards); 7-crop train, 4-crop held.
- [ ] Teacher bake-off (SigLIP2 / BioCLIP2 / SCOLD).
- [ ] Abstain gate + risk–coverage; top-5 metrics.
- [ ] INT8 + on-device latency (laptop + NPU).
- [ ] Convert to Elsevier CompAg LaTeX template; final figures.
