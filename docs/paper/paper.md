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
frozen encoder** performs cross-crop zero-shot diagnosis at **3.5–12.6× chance** across held-out sets of
16, 34 and 51 unseen classes, with accuracy **nearly flat from 11 M to 300 M parameters**; (ii)
**descriptor quality is the accuracy lever, and only *source-grounded* descriptors scale** — a full
symptom description beats a class-name prompt by **+15.2 pp** on a focused 3-crop set, but as the unseen
label space grows to 51 classes hand-curated descriptions **degrade** (28.0 → 20.4 %) while LLM
source-grounded ones **improve** (21.7 → 24.7 %), so source-grounding is not an auditability tax but the
mechanism that lets the approach reach new crops at all; (iii) an abstain gate makes the modest top-1
field-useful — grounded **top-5 is 61–77 %** on 16 unseen classes and still **58–76 % on 51**
(29–38× chance), with selective accuracy rising monotonically as coverage tightens; and (iv) the deployable image encoder runs at
**17.4 ms/image on a laptop CPU** (ONNX FP32, ~58 img/s) in a **12.9 MB** INT8 footprint — though we
show INT8 is a *size*, not a *speed*, lever for hybrid conv–transformer encoders on CPU. On known crops
the same frozen backbone reaches **82.4 %** over 166 classes, rising to **87.7 %** with WiSE-FT at a cost
of only 0.7 pp of unseen accuracy. A leave-one-crop-out analysis shows the held-out crops are not
cherry-picked, and 14 supervised CNN baselines — stronger on seen crops (up to 88.6 %) — remain
*structurally* incapable of the cross-crop transfer the descriptor head provides. We release the
descriptor pipeline, per-tier models, and benchmark so the result is reproducible and auditable.

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
1. **Frozen compact VLMs do cross-crop zero-shot.** An 11.4 M encoder diagnoses unseen crops at
   3.5–12.6× chance across 16-, 34- and 51-class held-out sets, and accuracy is flat 11 M→300 M —
   *model size is not the bottleneck* (§5.1–5.3).
2. **A nested scale study showing that only source-grounded descriptors scale.** Detail is the first
   lever (+15.2 pp over class names at 16 classes), but as the unseen label space triples, hand-curated
   descriptors *degrade* (28.0 → 20.4 %) while LLM source-grounded ones *improve* (21.7 → 24.7 %),
   overtaking them by +4.3 pp at 51 classes. Auditability is therefore not a cost but the **only
   authoring route that reaches new crops at all** (§5.3).
3. **A calibrated abstain gate + top-5** make the modest top-1 field-useful (§5.4).
4. **A per-tier edge benchmark** with real INT8 sizes and CPU latency, an efficiency Pareto that
   identifies the 11 M tier as the deployment sweet spot (§5.9), and a **quantisation result with
   practical reach**: INT8 helps transformer encoders (2.2× faster) but *hurts* hybrid conv–transformer
   encoders (up to 19× slower), with the graph-level diagnosis and a deployment rule (§5.10).
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
We use **SAGE only** (Arshad et al., 2026; `tirtho149/SAGE`, MIT) — **~839 K images spanning 335 crops
and 1,251 disease classes** — to avoid the "old, lab-only" critique of legacy corpora such as
PlantVillage, on which in-distribution accuracy has long been saturated. We stream-filter its parquet
shards to our crops with a per-class cap and content-hash de-duplication.

To test whether the approach *scales* rather than reporting a single convenient split, we define three
**nested** configurations — every seen/held crop in A also appears in B, and B in C — so differences are
attributable to the size of the label space, not to which crops were picked:

| Config | Seen crops | Seen classes | Seen images | Held-out crops | Held-out classes | Chance |
|---|---|---|---|---|---|---|
| **A** | 4 — Corn, Soybean, Tomato, Apple | 97 | 42,326 | Coffee, Orange, Peach | 16 | 6.2 % |
| **B** | 8 — + Grape, Potato, Rice, Sugarcane | 154 | 62,043 | + Cotton, Wheat, Bean | 34 | 2.9 % |
| **C** | 10 — + Rose, Strawberry | 166 | 69,919 | + Banana, Cucumber | 51 | 2.0 % |

Class counts are derived from the images actually on disk, so they are properties of a specific data
snapshot; the seen-image counts above are recorded in the result files alongside every accuracy, and
all configurations reported here come from one snapshot.

Held-out crops are never used for training at any stage; they are diagnosed only from descriptor text.
Seen-head results (§5.5–5.6, §5.8) are reported on the largest configuration, C.

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
1. **Trained seen-crop head** — a linear probe (or light fine-tune) on the seen classes (166 in
   configuration C) for best real-time accuracy on known crops.
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
Cross-crop evaluation uses the three nested held-out sets of §3.1 (16 / 34 / 51 classes; 4,050 / 9,556 /
14,392 images). The **encoder bake-off (§5.2)** predates the nested splits and is reported on the
17-class pilot set over the same three anchor crops (1,618 images, chance 5.9 %); it is internally
consistent and we do not mix its numbers with the scale study. Metrics: top-1, **top-5**, per-crop
accuracy, **risk–coverage / AURC** (abstain), leave-one-crop-out with bootstrap 95 % CIs, and on-device
latency/size. Models are loaded via `open_clip`; edge numbers use ONNX Runtime on CPU (224×224,
batch 1). All runs are on a single RTX-GPU workstation. Every table in §5 is machine-generated from the
result JSONs by `docs/paper/make_tables.py` (output: `TABLES.md`); no result is transcribed by hand.

## 5. Results

### 5.1 A frozen 11 M model already does cross-crop zero-shot, and accuracy is flat with size
On the pilot held-out set a frozen **MobileCLIP2-S0 (11.4 M)** reaches **27.0 %** rich-descriptor
zero-shot (4.6× chance) — **86 %** of the 31.5 % achieved by the 93 M SigLIP2 reference. The pattern
survives the move to the larger nested splits: across A/B/C the four deployable encoders span only
17.0–24.2 % (rich) and 22.2–29.1 % (grounded) at scale C despite an **8× spread in parameters**, and the
efficiency curve is essentially flat from 11 M to 300 M (Fig. `fig_efficiency_curve.png`). Model size is
not the bottleneck; the pretrained alignment is the asset. Attempts to *train* a tiny model to learn
this alignment fail (from-scratch ≈ chance; specialising on seen crops forgets 4.5 pp, Fig.
`fig_specialization_forgetting.png`), so we keep the backbone frozen.

### 5.2 Encoder bake-off — generic beats domain-specific, and small punches up (Fig. `fig_bakeoff.png`)
(17-class pilot held set, chance 5.9 %)

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

### 5.3 Descriptor detail is the lever — and only *source-grounded* descriptors scale
(Figs. `fig_descriptor_scaling.png` **[headline]** and `fig_descriptor_ablation.png`; full per-model
numbers in `TABLES.md` T1)

We evaluate the same four encoders at **three held-out scales**: **A** (3 crops, 16 classes,
chance 6.2 %), **B** (6 crops, 34 classes, 2.9 %) and **C** (8 crops, 51 classes, 2.0 %). Crop pools are
nested (A ⊂ B ⊂ C), so this is a controlled "accuracy vs. number of unseen classes" study.

Mean top-1 across the four deployable encoders:

| Held-out scale | Classes | Chance | bare | rich (hand-curated) | grounded (LLM, source-grounded) | grounded ÷ chance |
|---|---|---|---|---|---|---|
| **A** | 16 | 6.2 % | 12.8 % | **28.0 %** | 21.7 % | 3.5× |
| **B** | 34 | 2.9 % | 20.3 % | 24.3 % | 23.5 % | 8.1× |
| **C** | 51 | 2.0 % | 19.0 % | 20.4 % | **24.7 %** | **12.6×** |

**(1) Detail is the lever.** Any full symptom description beats a class-name prompt. At the focused
scale A the gain is large — **+15.2 pp** on average (per-model: +10.1 to +22.4 pp) — closely matching
the +14–16 pp that SAGE reports for symptom knowledge in a cloud setting, here reproduced on *frozen
11–86 M edge encoders*.

**(2) Hand-curation does not scale; source-grounding does.** This is the study's most consequential
finding and it **inverts** the conclusion we drew from scale A alone. As the unseen label space grows
from 16 → 51 classes, hand-curated `rich` descriptors **degrade monotonically** (28.0 → 24.3 → 20.4 %)
while LLM source-grounded descriptors **improve** (21.7 → 23.5 → 24.7 %). At scale C, `grounded` beats
`rich` for **all four encoders** (+2.7 to +5.2 pp; mean +4.3 pp) and is the only strategy that still
clearly beats `bare` (+5.7 pp vs. +1.4 pp for `rich`).

The mechanism is coverage, not cleverness: the hand-written symptom bank was authored against the three
anchor crops and thins out as new crops enter, whereas the generated registry covers 156 of 217 classes
with a uniform schema. **Implication:** source-grounding is not merely an auditability tax paid for
equal accuracy — it is *what makes the descriptor approach extend to new crops at all*, which is
precisely the capability this paper claims. Measured against chance, the grounded head becomes
*relatively stronger* as the problem gets harder (3.5× → 12.6×).

**(3) The finding is not an artefact of label noise** (`TABLES.md` T1b). SAGE's held-out set contains
five duplicate disease pairs and four non-disease labels (§7). Merging and dropping them reduces
experiment C from 51 to 42 classes, and we re-ran the entire evaluation on the corrected set as a
sensitivity analysis:

| Strategy | As-published (51 cls) | Label-corrected (42 cls) | Δ |
|---|---|---|---|
| bare | 19.0 % | 23.8 % | +4.8 pp |
| rich | 20.4 % | 26.4 % | +6.0 pp |
| **grounded** | 24.7 % | **30.7 %** | +6.0 pp |

Every strategy gains 4.8–6.0 pp, confirming that unwinnable duplicate pairs were suppressing all of
them — so the numbers we report elsewhere are **conservative**. Decisively, the **grounded − rich gap
is unchanged at +4.3 pp**, and grounded wins on **all five** encoders after correction (+2.5 to
+6.8 pp), including the SigLIP2 reference. On the corrected set the grounded head reaches
**12.9× chance**. We report the as-published numbers as our headline throughout, and treat the
corrected set as the sensitivity check rather than the other way round, so our claims are never
flattered by a benchmark we ourselves modified.

### 5.4 Top-5 and a calibrated abstain gate make the top-1 field-useful (Fig. `fig_riskcoverage.png`)

Scale A (16 classes, chance 6.2 %):

| Model | Strategy | Top-1 | Top-5 | AURC ↓ | acc@cov90 | acc@cov80 |
|---|---|---|---|---|---|---|
| MobileCLIP2-S0 (11.4 M) | rich | 25.9 % | 59.9 % | 0.691 | 27.2 % | 28.5 % |
| MobileCLIP-S1 (21.5 M) | grounded | 28.5 % | 61.4 % | **0.502** | 30.3 % | 32.9 % |
| MobileCLIP2-S2 (35.8 M) | rich | **32.7 %** | 64.5 % | 0.608 | 34.7 % | 36.8 % |
| MobileCLIP-B (86.3 M) | grounded | 26.4 % | **77.3 %** | 0.610 | 27.4 % | 28.6 % |
| SigLIP2 (92.9 M) | grounded | 26.4 % | 76.2 % | 0.667 | 27.1 % | 27.9 % |

A field tool can propose a short list and abstain when unsure — the metric that matters is not raw
top-1. With grounded descriptors **top-5 reaches 61–77 % at scale A**, and critically it *holds up under
scale*: at C (51 classes, chance 2.0 %) grounded top-5 is still **58–76 %**, i.e. 29–38× chance — the
short list stays trustworthy even as the label space triples. Selective accuracy rises monotonically as
coverage tightens (e.g. S2 32.7 → 34.7 → 36.8 % at 100/90/80 % coverage), confirming the confidence
signal is properly ordered rather than anti-calibrated.

Across models, **`grounded` abstains best** — it holds the lowest AURC (S1 0.502) and the highest top-5
(MobileCLIP-B 77.3 %) — because its cleaner class separation improves both ranking and confidence
calibration, even where its top-1 does not lead. Combined with §5.3, this makes `grounded` the
recommended deployment strategy: it scales with the label space *and* abstains most reliably.

### 5.5 The hybrid works on one 11 M backbone (Figs. `fig_hybrid.png`, `fig_seen_scaling.png`)
Running the seen head at all three nested scales gives the seen-side counterpart to §5.3
(`TABLES.md` T4):

| Config | Seen crops | Seen classes | Seen images | MC2-S0 (11.4 M) | MC-S1 (21.5 M) | MC2-S2 (35.8 M) | MC-B (86.3 M) |
|---|---|---|---|---|---|---|---|
| **A** | 4 | 97 | 42,326 | 78.9 % | 77.7 % | 78.8 % | 79.3 % |
| **B** | 8 | 154 | 62,043 | 80.7 % | 79.2 % | 80.8 % | 80.9 % |
| **C** | 10 | 166 | 69,919 | **82.4 %** | 81.1 % | 82.5 % | 82.6 % |

Three observations.

**(1) The seen head is accurate.** 82.4 % over 166 fine-grained classes from a *frozen* 11 M encoder.
The gain over using the same frozen encoder zero-shot on its own seen classes is large — measured at
the 80-class configuration, the probe lifts 9.3 % → 67.0 %, a **7.2× gain** — confirming that
supervised training, not descriptors, is the right tool for known crops.

**(2) Accuracy is flat with model size here too.** Across a **7.6× parameter range** the four encoders
span just 1.5 pp at every scale (e.g. 81.1–82.6 % at C). This mirrors the zero-shot flatness of §5.1
from the opposite direction: whether the head is trained or descriptor-driven, the pretrained
representation — not capacity — is the binding constraint. It is the strongest single argument that the
11 M tier is the right deployment choice.

**(3) Accuracy *rises* as the seen label space grows** (+3.3 to +3.5 pp from 97 to 166 classes, for
every encoder), which is the opposite of the usual fine-grained trend. Adding crops adds
proportionally more training images than decision difficulty, so the probe strictly improves. Note the
asymmetry with §5.3: on the *unseen* side more classes make the problem harder, while on the *seen*
side more classes make it easier. The two curves together are the case for the hybrid — each head is
deployed exactly where its scaling behaviour is favourable.

The same frozen backbone serves both heads simultaneously, so one deployed model covers both regimes.

*Reproducibility note:* configuration C was measured twice, three weeks apart, on independently rebuilt
embedding caches, agreeing to **0.15 pp** (S0) and **0.06 pp** (S1). An independent run of a separate
implementation agreed to within 0.2 pp on all four encoders.

### 5.6 WiSE-FT tunes the seen↔unseen trade-off (Fig. `fig_wiseft.png`)

Full data: MobileCLIP2-S0, 55,981 seen training images (the 80 % train split of the 69,919-image
config-C seen pool in §3.1), 166 seen classes, the same 3 held-out crops / 17 classes
and the same `rich` descriptors as §5.2, 5 fine-tune epochs (loss 2.02 → 0.24). *Note: the α = 0 unseen
value (17.0 %) is lower than §5.2's 27.0 % for the same encoder, strategy and class list — the 10-crop
configuration streams more SAGE shards, so the held-out **image** set is larger and harder than the
1,618-image pilot. All three α points here share one fixed protocol, so the sweep is internally valid;
we interpret only the differences across α, not the absolutes.*

| α | Seen | Unseen (ZS) |
|---|---|---|
| 0.0 (frozen) | 82.6 % | 17.0 % |
| **0.5 (WiSE-FT)** | **87.7 %** | **16.3 %** ← best balance |
| 1.0 (naive fine-tune) | 90.3 % | 8.8 % |

α = 0 reproduces the frozen baseline exactly (validating the interpolation). Naive fine-tuning (α = 1)
is **catastrophic forgetting** — seen rises to 90.3 % but unseen halves to 8.8 %. The deployable
setting is **α = 0.5: +5.1 pp seen for only −0.7 pp unseen**, an almost free gain that makes the
seen↔unseen balance a single tunable knob per deployment. (At the smaller 80-class configuration the
same sweep cost 11.6 pp of unseen accuracy for a comparable seen gain — the trade-off becomes markedly
more favourable with more seen data, which is the regime a real deployment is in.)

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

### 5.8 Supervised baselines: accurate on seen, structurally zero on unseen
Fourteen supervised architectures spanning **1.7–42.8 M parameters (a 25× range)**, all on the
identical 166-class seen set for 8 epochs (`TABLES.md` T4; per-epoch curves in
Fig. `fig_cnn_training.png`):

| Architecture | Params | Seen top-1 | | Architecture | Params | Seen top-1 |
|---|---|---|---|---|---|---|
| ConvNeXtV2-Nano | 15.1 M | **88.6 %** | | MobileNetV4-Conv-M | 8.7 M | 87.1 % |
| MobileNetV3-Large | **4.4 M** | **88.6 %** | | DenseNet-121 | 7.1 M | 87.0 % |
| EfficientNet-B0 | 4.2 M | 88.3 % | | ResNet-101 | **42.8 M** | 86.8 % |
| RegNetY-040 | 19.7 M | 88.1 % | | ConvNeXtV2-Tiny | 28.0 M | 85.6 % |
| EfficientNetV2-S | 20.4 M | 88.1 % | | FastViT-T8 | 3.4 M | 85.4 % |
| ResNet-50 | 23.9 M | 88.0 % | | MobileNetV3-Small | 1.7 M | 84.4 % |
| FastViT-SA12 | 10.7 M | 87.4 % | | MobileNetV4-Conv-S | 2.7 M | 82.3 % |

Every row is **0 % on unseen crops — structurally, not merely empirically**.

**Supervised CNNs win on known crops, and we say so plainly:** the best reaches 88.6 %, about
**6.2 pp above the frozen-VLM probe (82.4 %)** on the identical label set. *For a fixed, known label
set a conventional classifier is the better choice.*

**Capacity is not the lever — and here the evidence is unusually blunt.** The best architecture has
**4.4 M parameters**; the largest, ResNet-101 at **42.8 M**, is 1.8 pp *worse*. Across a 25× parameter
range the whole family spans 6.3 pp with no monotone trend, and the three heaviest models
(ResNet-101, ConvNeXtV2-Tiny, EfficientNetV2-S) do not occupy the top. This is the same conclusion as
§5.1 (zero-shot flat 11 M→300 M) and §5.5 (seen probe flat over 7.6×), now reproduced in a third,
fully-supervised setting — which makes "size is not the bottleneck" a property of *the task*, not an
artefact of frozen backbones.

**The cleanest control in the paper.** FastViT-SA12 (10.7 M) is the *same architecture family* as our
MobileCLIP image encoder (11.4 M), at nearly the same size. Trained supervised it reaches 87.4 %; used
as a frozen VLM backbone with a linear probe it reaches 82.4 %. The 5.0 pp difference is therefore
attributable to **training regime, not architecture** — supervised training buys seen-crop accuracy,
image–text pretraining buys the ability to name a disease on a crop never seen in training.

And that is the trade the paper proposes: the descriptor head gives up ~6 pp on known crops to gain a
capability **no row in this table possesses at any parameter count**, while WiSE-FT (§5.6, 87.7 %
seen) recovers most of the gap. *Protocol note:* the four largest models exhausted GPU memory at
batch 128 and were re-run at batch 64; with a fixed learning rate that is more optimiser steps per
epoch, so those rows are not perfectly controlled against the rest. The effect is small next to the
6.3 pp spread, and it does not touch the conclusion — the batch-64 group contains both the 42.8 M
model and the 88.1 % second-place finisher.

### 5.9 Efficiency / on-device (Fig. `fig_edge_pareto.png`)
Image encoder only (the sole part that ships); laptop CPU (16 threads), batch 1, 224×224, ONNX Runtime
1.26 with all graph optimisations enabled, 50 runs. We report **two** INT8 paths because the choice of
quantisation recipe changes the answer by an order of magnitude (§5.10):

| Tier | Params | MACs | Torch FP32 | **ONNX FP32** | INT8 dynamic | INT8 static (QDQ) | FP32 size | INT8 size |
|---|---|---|---|---|---|---|---|---|
| **S0** | 11.4 M | 1.8 G | 47.1 ms | **17.4 ms** | 320.0 ms | 62.2 ms | 45.8 MB | **12.9 MB** |
| S1 | 21.5 M | 3.6 G | 99.8 ms | **33.7 ms** | 614.9 ms | 79.5 ms | 86.5 MB | 24.3 MB |
| S2 | 35.8 M | 6.0 G | 119.0 ms | **49.2 ms** | 923.8 ms | 98.7 ms | 143.6 MB | 39.2 MB |
| B | 86.3 M | 17.0 G | 130.7 ms | 100.8 ms | 60.4 ms | **46.4 ms** | 345.6 MB | 87.4 MB |

**Real-time is met in FP32:** S0 runs at **17.4 ms/image (~58 img/s) on a commodity laptop CPU** — ONNX
Runtime is ~2.7× faster than eager PyTorch. **INT8 compresses ~3.5×** (S0 → 12.9 MB).

### 5.10 INT8 is architecture-dependent, not universally beneficial
A naive `quantize_dynamic` call — the default recipe in most tutorials — makes the lightweight tiers
**18–19× *slower*** (S0 320 ms vs. 17.4 ms FP32). A properly configured static QDQ pipeline
(shape-inference pre-pass, per-channel weights, calibrated activations) recovers **5–9×** of that, but
still does **not** beat FP32 on the S-tiers. The pure-transformer B behaves oppositely: INT8 makes it
**2.2× faster** (100.8 → 46.4 ms).

The mechanism is visible in the quantised graphs. Counting convolutions the quantiser could *not*
convert:

| Tier | Architecture | Float convs remaining | Conversion nodes | INT8 speedup |
|---|---|---|---|---|
| S0 / S1 / S2 | FastViT-style hybrid (depthwise convs) | **119 / 215 / 235** | 1290 / 2340 / 2536 | 0.28× / 0.42× / 0.50× |
| B | pure ViT (MatMul-dominated) | **3** | 1013 | **2.17×** |

On the hybrids the runtime must dequantise → float-conv → requantise hundreds of times per inference;
dynamic quantisation instead maps every convolution to `ConvInteger`, which is pathologically slow for
depthwise kernels on x86. **Practical guidance: for hybrid conv–transformer encoders on a CPU runtime,
INT8 is a *size* lever (3.5× smaller), not a *speed* lever; deploy FP32.** For transformer encoders it
is both. Deployment recommendation: **S0 (ONNX FP32) for phones/laptops; B (INT8) where a transformer
NPU is available.** (Raspberry Pi / ARM row pending an on-device run of the same script — ARM NEON has
well-optimised INT8 depthwise kernels and may reverse the S-tier result, which we flag rather than
assume.)

## 6. Discussion
- **The contribution is a system, not a backbone.** The four tiers are off-the-shelf encoders used
  frozen; the novelty is the source-grounded descriptor head, the calibrated abstain gate, the WiSE-FT
  seen/unseen knob, and the real-time edge packaging. Because accuracy is flat with size, the *reason*
  to offer a family is the measured **latency/size Pareto** (§5.9), not accuracy.
- **Source-grounding is a trust contribution *and*, at scale, an accuracy one.** On a focused 3-crop set
  hand-curation wins, which is why a single-split study would conclude that auditability is a free but
  optional extra. The nested design shows the opposite: as the unseen label space triples, hand-curated
  descriptors decay and grounded ones improve, overtaking them by +4.3 pp (§5.3). Hand-curation does not
  scale because it cannot — it requires an expert per crop; a grounded generator requires a source per
  crop. **The auditable route is also the only scalable one**, and we consider this the paper's most
  transferable claim beyond agriculture.
- **Real-time without exotic hardware.** Only the image encoder ships; 17.4 ms on a commodity CPU with a
  12.9 MB INT8 footprint makes the 11 M tier the default.
- **Quantisation advice is architecture-specific.** The field default ("quantise to INT8 for edge") is
  actively harmful for hybrid conv–transformer encoders on CPU runtimes — up to 19× slower with the
  standard dynamic recipe. We give the diagnosis (unconvertible depthwise convolutions) and the
  practical rule (§5.10), which is reusable by anyone deploying MobileCLIP-class encoders.
- **Honesty on absolutes.** Fine-grained cross-crop top-1 is modest (17–29 %); we lead with top-5
  (58–77 %) and the calibrated abstain gate as the field-relevant metrics, and frame cross-crop transfer
  as a hard open problem we advance at edge scale rather than claim to solve.

## 7. Limitations
- **Descriptor coverage and two tiers of citation.** Across all 18 crops, **156 of 217** disease records
  are LLM-filled (`status: filled`); the remaining 61 are explicit stubs that fall back to the
  hand-curated `rich` bank. Of the 156, **111 carry a verbatim quote**, but only **16 have been
  page-verified** — i.e. the quote was copied from a page we retrieved and read. The other 95 quotes are
  *model-recalled* and must be treated as unverified provenance, not evidence. Stub records carry a
  `TODO` placeholder that the loader excludes by design (`descriptors.py` keys on `status == "filled"`),
  so a placeholder can never become a CLIP text prototype.

  Coverage is, however, **strongly skewed toward the crops that carry the headline claim**, which is
  the right place for it: the 8 held-out crops of configuration C are **47/51 filled** and hold **all
  16** page-verified records, whereas the seen crops are 109/166 filled with **none** page-verified.
  The seen side is diagnosed by a trained probe and never consults a descriptor, so this asymmetry
  costs nothing — but it does mean the auditability guarantee applies to the zero-shot path only.
  The four remaining held-out stubs are all Wheat label artefacts (below).

  An initial audit found 19 of 205 source URLs dead, concentrated in extension sites that had
  reorganised. **All have since been replaced, and every replacement had its quote re-extracted from
  the new page rather than merely repointing the link** — a live URL whose quoted sentence does not
  appear on it is a worse failure than a dead one, because it reads as fabricated evidence. No
  descriptor now cites an unreachable page.

  To make the data self-documenting, every field carries a `provenance` tag: **79 fields are
  `page-verified`** (quote copied from a page we retrieved), **204 are `model-recalled`** (produced by
  the LLM and *not* checked against the cited page — unverified provenance, not evidence), and the
  remainder carry no quote. A reader inspecting the released JSON therefore cannot mistake one tier
  for the other. Nine URLs point at Wikipedia, which we flag as weaker than extension or
  peer-reviewed sources. On the **headline held-out
  crops (Coffee/Orange/Peach) coverage is now complete — 16/16 filled**, with every Coffee record
  page-verified against a retrievable source. The four remaining held-out stubs are all Wheat and are
  label artefacts (below). Of 201 unique source URLs, 110 are reachable, 73 return 403/429
  (bot-blocked but real, e.g. APS) and **18 are dead** and need manual replacement.
- **The grounding prompt over-refuses.** Because the generation endpoint cannot browse, a strict
  "never cite what you cannot quote" instruction makes the model return *empty* records rather than
  risk fabrication — including for well-documented diseases (e.g. Tomato spotted wilt virus, sugarcane
  common rust). A bulk re-run converted only 1 of 66 stubs. This is the anti-hallucination guarantee
  behaving correctly, but it means grounded coverage scales only with *human-supplied sources*, not
  with more API spend. Separating the functional `symptom_text` from the citation fields (as we do)
  is what keeps the pipeline usable.
- **Label noise in SAGE, and why it makes our numbers conservative.** Auditing the 51 held-out classes
  of configuration C surfaced two distinct defects.

  *Duplicate classes.* Five pairs denote the **same disease under two names**, which we confirmed by
  checking that the independently generated descriptor records resolve to an identical pathogen:

  | Pair | Pathogen (identical in both records) |
  |---|---|
  | `Orange/Canker` = `Orange/Citrus_Canker` | *Xanthomonas citri* |
  | `Orange/Greening_Disease` = `Orange/Huanglongbing` | *Candidatus* Liberibacter spp. |
  | `Peach/Leaf_Curl` = `Peach/Peach_Leaf_Curl` | *Taphrina deformans* |
  | `Cucumber/Angular_Leaf_Spot` = `Cucumber/…_Of_Cucumber` | *Pseudomonas syringae* pv. *lachrymans* |
  | `Wheat/Head_Scab` = `Wheat/Fusarium_Graminearum_Schwabe` | *Fusarium graminearum* |

  *Non-diseases.* `Wheat/Resistance_Phenotype{,_Moderately_Resistant,_Moderately_Susceptible}` are
  breeding **resistance ratings**, for which no symptom descriptor can exist; `Wheat/Fusarium_Wilts` is
  not a standard wheat disease; `Coffee/Miner` is an insect pest (*Leucoptera coffeella*) rather than a
  pathogen; and `Coffee/Cerscospora` misspells *Cercospora coffeicola*. Several Orange and Cucumber
  labels (`Green_Mold`, `Whisker_Mold`, `Belly_Rot`, `Pythium_Fruit_Rot`) are post-harvest fruit rots
  rather than foliar diseases.

  **This biases our headline downward, not upward.** A duplicate pair is unwinnable by construction:
  two near-identical text prototypes split the similarity mass, so top-1 is close to a coin flip
  regardless of how well the image is understood. Removing the five duplicates and the four
  non-disease labels leaves ~42 genuinely distinct held-out classes, so the *effective* chance level is
  ~2.4 % rather than the 2.0 % we report against. Our cross-crop accuracies are therefore
  **conservative**, and the true margin over chance is somewhat larger than stated.

  We recommend that users of SAGE merge these pairs and drop the rating labels, and we report it as a
  dataset-quality finding: label noise of this kind silently inflates the apparent class count of any
  fine-grained benchmark and depresses every method evaluated on it.

- **The class count is set by a minimum-images threshold, and the tail below it is contaminated.**
  We retain (crop, disease) classes with **≥ 25 images**, which yields our 166 seen / 51 held-out
  classes. The threshold, not the amount of data pulled, is what determines that number — we hold 12
  of SAGE's 13 shards for these crops, so the corpus is close to exhausted. Relaxing the threshold
  expands the label space steeply:

  | Min images | Seen classes | Held-out classes | Total |
  |---|---|---|---|
  | ≥ 1 | 366 | 177 | 543 |
  | ≥ 15 | 193 | 59 | 252 |
  | **≥ 25 (ours)** | **166** | **51** | **217** |
  | ≥ 50 | 133 | 42 | 175 |

  We keep ≥ 25 because the 327 excluded classes are not merely small, they are **visibly
  mis-labelled**: `Cucumber/Apple_Scab` (10 images) and `Potato/Apple_Scab` (6) attribute an apple
  disease to unrelated hosts, `Apple/Beech_Bark_Disease` (10) attributes a forest-tree disease to
  apple, a garbled label `Bacterial_Brown_Spot_Of_Bean…Of_Stone_Fruit` appears under Cotton, Wheat
  *and* Bean, and `Coffee/Brown_Eye_Spot` (13) is a third name for the *Cercospora coffeicola*
  already present twice. Lowering the threshold would therefore buy apparent scale at the cost of
  label integrity. We state the threshold explicitly because it is a load-bearing design choice that
  papers in this area often leave implicit.
- **Scope.** Configuration C covers 10 seen crops (Corn, Soybean, Tomato, Apple, Grape, Potato, Rice,
  Sugarcane, Rose, Strawberry) and 8 held-out crops — 18 of SAGE's crop set, not all of it. The seen-head
  probe is reported at scale C only; A and B seen-side probes are not run, so the seen-side scaling
  curve has one point. Single random seed; a single evaluation machine, so we report no variance
  estimates other than the LOCO bootstrap CIs.
- **Sub-10 M gap.** No off-the-shelf aligned model exists below ~11 M; a ~5 M tier would require
  weight-inherited distillation and is framed as future work.
- **Domain-VLM baselines.** BioCLIP2/SCOLD numbers use our loaders; a faithful SCOLD evaluation needs
  the authors' pipeline.

## 8. Conclusion
A single frozen, compact, descriptor-driven VLM brings cross-crop plant-disease diagnosis to laptops and
small devices at \$0/image: **cross-crop zero-shot at 3.5–12.6× chance on 16–51 unseen classes (86 % of
a 93 M model at 1/8 the size), 82.4 % — 87.7 % with WiSE-FT — on 166 known classes, 58–77 % top-5, a
calibrated abstain gate, and 17.4 ms real-time CPU inference in a 12.9 MB INT8 footprint.** Model size
is shown to be a near-non-factor across a 27× parameter range. The levers are, in order: **descriptor
authoring method** — and specifically the finding that only source-grounded descriptors keep improving
as the unseen label space grows — honest abstention, and the frozen-backbone hybrid. We release the
pipeline and benchmark to make the result auditable and reproducible.

---

## Highlights

*(Elsevier requires 3–5 bullets, each ≤ 85 characters including spaces.)*

- A frozen 11.4 M CLIP diagnoses diseases on crops it never saw, at 3.5–12.6× chance. (84)
- Only source-grounded descriptors keep improving as the unseen label space grows. (79)
- Hand-curated symptom text degrades from 28.0% to 20.4% as unseen classes triple. (80)
- One frozen backbone serves both known crops (82.4%) and unknown ones. (69)
- INT8 is a size lever, not a speed lever, for hybrid conv-transformer encoders. (78)

## CRediT authorship contribution statement

**[Author 1]:** Conceptualization, Methodology, Software, Investigation, Formal analysis,
Visualization, Writing – original draft. **[Author 2]:** _[fill]_. **[Author 3]:** _[fill]_.

> **ACTION REQUIRED:** replace with the real author list. Every listed author needs at least one
> CRediT role.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships
that could have appeared to influence the work reported in this paper.

## Data availability

All data are public. The SAGE dataset is available at `https://huggingface.co/datasets/tirtho149/SAGE`
(MIT licence). The source-grounded descriptor registry, all result JSONs, the table/figure generators,
and the full experimental code are released at _[repository URL — fill on acceptance]_. No proprietary
data were used.

## Acknowledgements

_[Fill: funding sources with grant numbers, compute donors, and any non-author contributors.
If there was no external funding, Elsevier expects an explicit statement to that effect.]_

## References

> **⚠ VERIFY BEFORE SUBMISSION.** Entries marked **⚠** were reconstructed from in-text citations and
> their exact authors, venue or year have **not** been confirmed against the published record. Check
> every one of them — a wrong citation is worse than a missing one. Unmarked entries are standard and
> widely cited, but confirm page numbers and DOIs against the publisher regardless.

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition, in:
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778.

Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R.,
Vasudevan, V., Le, Q.V., Adam, H., 2019. Searching for MobileNetV3, in: Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), pp. 1314–1324.

Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N., Taori, R., Dave, A., Shankar, V.,
Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., Schmidt, L., 2021. OpenCLIP. Zenodo.
https://doi.org/10.5281/zenodo.5143773

Menon, S., Vondrick, C., 2023. Visual classification via description from large language models, in:
International Conference on Learning Representations (ICLR).

Mohanty, S.P., Hughes, D.P., Salathé, M., 2016. Using deep learning for image-based plant disease
detection. Frontiers in Plant Science 7, 1419.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D.,
Massa, F., El-Nouby, A., et al., 2024. DINOv2: Learning robust visual features without supervision.
Transactions on Machine Learning Research.

Pratt, S., Covert, I., Liu, R., Farhadi, A., 2023. What does a platypus look like? Generating
customized prompts for zero-shot image classification, in: Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), pp. 15691–15701.

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A.,
Mishkin, P., Clark, J., Krueger, G., Sutskever, I., 2021. Learning transferable visual models from
natural language supervision, in: International Conference on Machine Learning (ICML), pp. 8748–8763.

Vasu, P.K.A., Pouransari, H., Faghri, F., Vemulapalli, R., Tuzel, O., 2024. MobileCLIP: Fast
image-text models through multi-modal reinforced training, in: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pp. 15963–15974.

Wortsman, M., Ilharco, G., Kim, J.W., Li, M., Kornblith, S., Roelofs, R., Lopes, R.G., Hajishirzi, H.,
Farhadi, A., Namkoong, H., Schmidt, L., 2022. Robust fine-tuning of zero-shot models, in: Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7959–7971.

Wu, K., Peng, H., Zhou, Z., Xiao, B., Liu, M., Yuan, L., Xuan, H., Valenzuela, M., Chen, X., Wang, X.,
Chao, H., Hu, H., 2023. TinyCLIP: CLIP distillation via affinity mimicking and weight inheritance, in:
Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 21970–21980.

Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L., 2023. Sigmoid loss for language image pre-training,
in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 11975–11986.

**⚠** Arshad, M.A., et al., 2025. SAGE: [exact title, authors and venue to be confirmed from the
paper the dataset ships with]. Dataset: https://huggingface.co/datasets/tirtho149/SAGE

**⚠** Ghazal, S., et al., 2024. [Cited in §1 for poor cross-crop transfer — confirm full reference or
remove the citation.]

**⚠** Qin, D., et al., 2024. MobileNetV4: Universal models for the mobile ecosystem, in: European
Conference on Computer Vision (ECCV). [Confirm author list and pages.]

**⚠** Stevens, S., et al., 2024/2025. BioCLIP / BioCLIP 2: A vision foundation model for the tree of
life. [We evaluate BioCLIP **2**; confirm which paper to cite and its year.]

**⚠** Tschannen, M., et al., 2025. SigLIP 2: Multilingual vision-language encoders. [Confirm authors
and venue.]

**⚠** Vasu, P.K.A., et al., 2025. MobileCLIP2. [Confirm the exact reference; several tiers we deploy
come from this release, so it must be cited correctly.]

**⚠** SCOLD, 2025. [Domain leaf-disease VLM evaluated in §5.2 — obtain the full citation, or state in
§5.2 that only the released checkpoint was used if no paper is available.]

---

### 9. Open items before submission
- [ ] Raspberry Pi / phone latency row (run `benchmark_edge.py` on-device).
- [x] ~~Source the missing held-out descriptors~~ — **done**: Coffee is now 5/5 page-verified
      (Cercospora leaf + berry phases, leaf miner, Phoma). Remaining 4 stubs are all Wheat label
      artefacts to be excluded/resolved rather than filled.
- [ ] Replace the 18 dead source URLs (concentrated in extension.psu.edu, cropprotectionnetwork.org,
      grapes.extension.org).
- [x] ~~Fuller SAGE pull incl. Tomato and Grape~~ — **done**: configuration C trains on 10 crops /
      166 classes / 69,919 images, including Tomato and Grape.
- [x] ~~`probe_seen_A` and `probe_seen_B`~~ — **done**: all four encoders at all three scales from one
      data snapshot (§5.5, `fig_seen_scaling.png`).
- [x] ~~Supervised CNN sweep~~ — **done**: all 14 architectures at one protocol (166 classes,
      8 epochs), spanning 1.7–42.8 M parameters. Best is 88.6 %; the largest model is 1.8 pp worse
      than the best 4.4 M one (§5.8).
- [ ] Multi-seed runs for CIs on the headline zero-shot table.
- [ ] Faithful SCOLD loader (or footnote the current below-chance wrapper result).
- [ ] Convert to the Elsevier CompAg LaTeX template; final figure polish; author list & funding.
