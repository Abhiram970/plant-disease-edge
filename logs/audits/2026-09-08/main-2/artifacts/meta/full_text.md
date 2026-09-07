## Compact Vision–Language Models for Cross-Crop Plant-Disease
## Diagnosis at the Edge
Pinni Venkata Abhirama,∗, Gaurav Shrivastavab and Rahul Ananthasayanamc
aIndependent Researcher, Hyderabad, Telangana, India
bIndependent Researcher, Pune, Maharashtra, India
cIndependent Researcher, Chennai, Tamil Nadu, India
A R T I C L E I N F O
Keywords:
foundation models
vision–language models
plant disease
cross-crop zero-shot
edge deployment
source-grounded descriptors
precision agriculture
Abstract
Cloud vision–language models diagnose plant disease accurately but bill per query and need
connectivity, suiting neither smallholder nor in-field use. The harder obstacle: a classifier cannot
name a disease on a crop absent from its training set. We ask how small a deployable model can be
while still diagnosing unseen crops, by matching a leaf image to symptom descriptions written by
a large language model and grounded in cited sources. Using the SAGE crop-disease dataset and
frozen compact contrastive language–image pre-training encoders, we report four results. First, frozen
encoders from 11.4 to 86.3 million parameters diagnose unseen crops at 2.2–13.9 times chance across
nested held-out sets of 16, 34 and 51 classes, accuracy nearly flat across that range. Second, descriptor
authoring is the central lever: source-grounded text improves from 21.5% to 23.9% as the unseen
label space triples. A matched ungrounded control isolates sourcing from mere per-class distinctness:
requiring a citable source is worth +1.96 points on average and helps every encoder, so auditability is
free. Third, an abstain gate makes top-1 field-useful: grounded top-5 reaches 60–77% on 16 unseen
classes and 59–75% on 51. Fourth, the encoder runs at 17.4 ms per image on a laptop processor at
45.8 MB, and 8-bit quantisation shrinks it to 12.9 MB at the cost of latency: for hybrid encoders INT8
is a size lever, not a speed one. On known crops the same backbone reaches 82.2% over 166 classes;
supervised CNN baselines are stronger there yet structurally incapable of cross-crop transfer.
1. Introduction
Cloud VLM diagnosers achieve high in-the-wild accuracy
but are metered per query and require connectivity — a
poor fit for smallholder and field use, which needs offline,
low-cost, real-time inference. A second, deeper obstacle is
generalisation: models trained on one crop transfer poorly
to others [3], and edge deployment additionally demands
compression. Cross-crop transfer is therefore the foundation-
model-native open problem this Special Issue targets: it
is enabled only by an image–text-aligned VLM, not by a
conventional classifier.
We study the question: how small can a model be and still
perform cross-crop disease zero-shot at the edge? Our system
is one frozen compact Contrastive Language–Image Pre-
training (CLIP) encoder with two heads and an abstain router:
a trained head for accurate real-time diagnosis on known
crops, and a descriptor head that diagnoses unseen crops by
matching image embeddings to large-language-model (LLM)
authored, source-grounded symptom-descriptor text. Only
the image encoder ships to the device; text prototypes are
precomputed offline.
Contributions
1. Frozen compact VLMs do cross-crop zero-shot.
Averaged over the four deployable tiers, they diagnose
unseen crops at 2.2–13.9× chance across 16-, 34- and
∗Corresponding author
abhiramp428@gmail.com (P.V. Abhiram);
gauravjshrivastava10@gmail.com (G. Shrivastava); tnarahul@gmail.com (R.
Ananthasayanam)
ORCID(s): 0009-0007-8783-7362 (P.V. Abhiram)
51-class held-out sets (the 11.4 M tier alone spans 2.9–
11.3×), and accuracy is flat from 11 M to 300 M —
model size is not the bottleneck (Sections 4.1–4.3).
2. A nested scale study isolating what descriptor
authoring actually buys. Symptom text beats a bare
class name at every scale (+8.8 points at 16 classes for
source-grounded text), and source-grounded descrip-
tors keep improving as the unseen label space triples.
We further show that the usual comparison against
a keyword-retrieved symptom bank is confounded
by prototype collisions rather than by grounding,
and separate the two with an ungrounded-generation
control (Section 4.3).
3. A margin-ranked abstain gate plus top-5 makes the
modest top-1 field-useful (Section 4.4).
4. A per-tier edge benchmark with real INT8 sizes and
CPU latency, and a quantisation result with practical
reach: INT8 helps transformer encoders but hurts hy-
brid convolution–transformer encoders, with a graph-
level diagnosis and a deployment rule (Section 4.8).
5. Rigorous negatives that save effort: naive fine-tuning
causes catastrophic forgetting, a supervised CNN
cannot transfer cross-crop, and a leave-one-crop-out
analysis shows the held-out crops are not cherry-
picked.
2. Related work
Agricultural VLMs SAGE [1] provides our data and the
cloud agent we contrast against; its {value, source_url,
verbatim_quote} source-grounded schema and its reported
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 1 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
14–16 point gain from symptom knowledge motivate our
descriptor design. We are the deployable edge counterpart at
$0/image.
Descriptor-based classification Descriptor-based CLIP
classification (DCLIP) [9] and Customized Prompts via
Language models (CuPL) [13] classify by LLM-written
visual descriptions. In contrast, our descriptors are source-
grounded and therefore auditable, compressed to the edge,
and evaluated for cross-crop agricultural transfer rather than
general retrieval.
Compact CLIPs CLIP [15] established image–text con-
trastive pretraining; MobileCLIP [17], its successor Mobile-
CLIP2 [4] and TinyCLIP [19] compress it, and provide our
frozen backbones. We use the open_clip [8] implementations
throughout, with SigLIP 2 [16], whose sigmoid objective
follows Zhai et al. [20], as a reference ceiling. We contribute
an agricultural cross-crop zero-shot study and an edge
efficiency analysis rather than a retrieval benchmark.
Robust fine-tuning WiSE-FT [18] explains our forget-
ting result and motivates the frozen-backbone plus weight-
ensembling design.
Domain foundation models We evaluate the biological
foundation model BioCLIP 2 [5] and the leaf-disease model
SCOLD [11] as candidate backbones and find they do not
transfer under our descriptor protocol (Section 4.2).
Vision-only foundation models Self-supervised encoders
such as DINOv2 [12] learn strong visual features but have
no text tower, so they cannot match an image against a
written symptom description. Zero-shot diagnosis of an
unseen crop therefore requires an image–text-aligned model
by construction, which is why every backbone we evaluate is
CLIP-family.
Supervised
baselines Our
comparison
set
spans
ResNet [6], MobileNetV3 [7] and MobileNetV4 [14] among
others (Section 4.6).
3. Materials and methods
3.1. Dataset and the nested design
We use SAGE [1] only, drawing images from the re-
leased dataset [2] — ∼839 K images over 335 crops and
1,251 disease classes, released under the MIT licence — to
avoid the “old, lab-only” critique of legacy corpora such as
PlantVillage [10], on which in-distribution accuracy has long
been saturated. We stream-filter shards to our crops with
content-hash de-duplication, storing images at a longest edge
of 288 pixels since every model here operates at 224. Classes
are capped at 600 images for held-out crops and 1,000 for seen
crops, and a class needs at least 25 images to enter the study
at all; images are taken in sorted filename order up to the cap.
Filenames in the released dataset are content hashes, so this
order is uncorrelated with capture time or any other property
of the image: the cap acts as a deterministic random sample
rather than favouring, say, the earliest-collected photographs,
and the subset is exactly rebuildable from the released code.
The caps bound class imbalance rather than save disk: they
are what keep the majority-class baseline at 4.2% on the
largest held-out configuration, so a frequency prior cannot
masquerade as transfer. Nine of the 51 held-out classes reach
the 600-image cap, so the held-out set is a capped sample of
the pinned revision rather than the whole of it. The resulting
subset is 84,123 images over 217 classes and 18 crops: 14,204
images in the 51 held-out classes and 69,919 in the 166 seen
classes.
We pin the dataset to revision bc9bd2899f (7 May 2026).
This is not a formality: SAGE was re-released on 24 August
2026 with re-canonicalised crop and disease names, and the
later release drops Cotton entirely — its own canonicalisation
record marks every Cotton entry as having no canonical
crop. Evaluating against the later release would therefore
yield seven held-out crops and 48 classes at our largest
configuration rather than the eight and 51 reported here. Every
number in this article refers to the pinned revision, and the
fetch code pins the same commit hash.
To test whether the approach scales rather than reporting a
single convenient split, we define three nested configurations
— every seen and held-out crop in A also appears in B, and B
in C — so differences are attributable to the size of the label
space and not to which crops were chosen. Configuration
A holds out Coffee, Orange and Peach (16 classes); B adds
Cotton, Wheat and Bean (34); C adds Banana and Cucumber
(51). Seen-set sizes are given in Table 3. Held-out crops are
never used for training at any stage; they are diagnosed only
from descriptor text.
3.2. Source-grounded descriptors
Each disease carries a symptom record with fields for
pathogen, affected organs and visual symptoms, each field
carrying a value, a source URL and a verbatim quote. Records
are generated by an LLM under a strict grounding prompt that
forbids un-cited claims, then audited. Because the generation
endpoint cannot browse, raw citations are model-recalled;
we therefore web-verify the headline held-out diseases,
replacing citations with sentences copied verbatim from
reachable authoritative pages. The symptom text — the string
that becomes the CLIP text prototype — is the functional
component; the verified citations provide the auditability
guarantee. Records that could not be grounded are stored as
explicit stubs and are excluded by the loader, so a placeholder
can never become a prototype.
3.3. Architecture
One frozen image–text encoder per tier, plus: (1) a trained
seen-crop linear probe; (2) a zero-shot descriptor head se-
lecting the nearest source-grounded text prototype by cosine
similarity; (3) WiSE-FT weight ensembling, interpolating
fine-tuned and frozen visual weights at ratio 𝛼; and (4) an
abstain router on the top-1 minus top-2 similarity margin.
Only the image encoder deploys. We evaluate four deployable
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 2 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
tiers (11.4, 21.5, 35.8 and 86.3 M image-encoder parameters)
plus a 92.9 M reference ceiling.
3.4. Descriptor strategies and the grounding
control
We compare four prompt strategies of increasing speci-
ficity, all evaluated with the same frozen encoders and the
same held-out splits. Bare uses the class name alone. Crude
adds one generic keyword sentence. Rich retrieves a hand-
curated symptom paragraph from a 32-key keyword bank.
Grounded uses the LLM-authored, source-cited symptom
text of Section 3.2, falling back to rich for any class without
a record.
Why rich cannot isolate grounding Comparing grounded
against rich conflates two different properties. Retrieval
assigns text by matching a disease name against 32 generic
keys, so distinct classes routinely receive identical prototypes:
at configuration C only 8 of the 51 held-out classes receive
a unique entry, 17 fall back to the bare class name, and 26
share text with another class, the largest groups being seven
classes keyed on spot and five on blight. Collisions grow with
the label space (2, 14 and 26 classes at A, B and C) while
coverage stays flat (68.8%, 61.8%, 66.7%). Any advantage
over this baseline therefore measures per-class distinctness
rather than sourcing. This is a limitation of symptom-bank
retrieval as we implemented it, not of per-class generation
methods such as DCLIP or CuPL, which emit text per class
and cannot collide.
The ungrounded control To isolate the sourcing constraint
we generate a matched ungrounded arm: the same model,
the same schema, the same temperature and the same seeds
as grounded, with only the requirement to cite a retrievable
source removed. Because the two arms differ in exactly one
instruction, the difference between them estimates the effect
of sourcing alone.
Paired-subset evaluation Coverage differs between the
arms, and a naive comparison would absorb that difference.
Since text_for falls back to the keyword bank wherever an
arm has no record, an arm covering 41 of 51 classes is four-
fifths its own text and one-fifth shared bank text, which dilutes
both arms toward a common baseline. We therefore evaluate
both arms on the intersection of classes that every seed of
both arms filled genuinely, so that the sourcing requirement is
the only difference between them. The intersection is smaller
than the full held-out set and chance rises accordingly; we
report both. Encoders serve as the replication unit: each of
the four deployable tiers is an independent test of the same
claim, and we report the per-encoder difference, the mean,
and a confidence interval over encoders.
4. Results
4.1. A frozen 11 M model already does cross-crop
zero-shot, and accuracy is flat with size
Across the nested splits (Protocol N) the four deploy-
able encoders span only 19.3–24.6% (rich) and 21.6–27.3%
(grounded) at configuration C despite a 7.6-fold spread in
parameters, and the efficiency curve is essentially flat from
11 M to 300 M (Fig. 1). Model size is not the bottleneck;
the pretrained alignment is the asset. Attempts to train a
small model to learn this alignment fail — on the earlier
17-class pilot set (Protocol P, not comparable with the nested
splits above) a 5.5 M student distilled from scratch reaches
only 11.0% against 5.9% chance, versus 19.9% for the frozen
11.4 M encoder it was distilled from; and specialising that
encoder on seen crops loses 4.5 points of unseen accuracy to
catastrophic forgetting (Fig. 2) — so we keep the backbone
frozen.
4.2. Encoder bake-off
Fig. 3 summarises the comparison. The 11.4 M encoder
recovers 86% of the 92.9 M reference at one eighth the
parameters, and beats the larger 21.5 M model — a training-
recipe effect that reinforces that pretraining quality, not
size, drives transfer. Biological and domain leaf-disease
foundation models perform at or below chance under a
descriptor protocol: taxonomy-style contrastive pretraining
does not align to symptom-descriptor text. We note openly
that our wrapper for one domain model is a best-effort load
and a faithful evaluation would need the authors’ inference
pipeline; we do not rest any claim on it.
4.3. Descriptor authoring is the lever
Table 1 and Figs. 4 and 5 carry the paper’s central result.
Detail is the first lever Any full symptom description
beats a class-name prompt. At the focused scale A, source-
grounded text gains 8.8 points on average over a bare class
name on frozen 11–86 M edge encoders.
Source-grounded descriptors keep improving with scale
As the unseen label space grows from 16 to 51 classes, source-
grounded descriptors improve monotonically (21.5%, 22.7%,
23.9%), and at configuration C they are the strategy that most
clearly beats a bare class name. The generated registry covers
156 of 217 classes under a uniform schema.
Sourcing itself carries a small, reproducible advantage
The matched ungrounded control of Section 3.4 isolates the
effect of requiring a citable source. On the 41 paired classes
at configuration C (11,337 images, chance 2.44%), averaged
over seven shared generation seeds, the source-grounded
arm leads the ungrounded arm on every deployable encoder:
+1.76, +2.00, +2.15 and +1.93 points for MobileCLIP2-
S0, MobileCLIP-S1, MobileCLIP2-S2 and MobileCLIP-B,
a mean of +1.96 points with a 95 % confidence interval of
[+1.70, +2.22] over encoders. The strength of this result is
the agreement across encoders rather than the width of that
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 3 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
Table 1
Cross-crop zero-shot accuracy at three held-out scales, by
descriptor strategy.
Model
Params (M)
bare
crude
rich
grounded
Config A — 16 unseen classes, chance 6.2%
MobileCLIP2-S0
11.4
13.8%
11.9%
26.1%
17.6%
MobileCLIP-S1
21.5
12.4%
17.5%
24.9%
28.6%
MobileCLIP2-S2
35.8
10.3%
10.4%
33.3%
13.7%
MobileCLIP-B
86.3
14.2%
16.7%
31.2%
26.2%
ViT-B-16-SigLIP2
92.9
17.1%
19.6%
30.4%
26.8%
mean
—
12.7%
—
28.9%
21.5%
Config B — 34 unseen classes, chance 2.9%
MobileCLIP2-S0
11.4
21.8%
17.5%
23.3%
20.5%
MobileCLIP-S1
21.5
17.5%
18.6%
23.6%
25.6%
MobileCLIP2-S2
35.8
19.0%
16.1%
24.8%
19.4%
MobileCLIP-B
86.3
23.1%
22.6%
28.4%
25.3%
ViT-B-16-SigLIP2
92.9
27.1%
26.2%
29.9%
27.3%
mean
—
20.3%
—
25.0%
22.7%
Config C — 51 unseen classes, chance 2.0%
MobileCLIP2-S0
11.4
20.2%
15.6%
19.3%
21.6%
MobileCLIP-S1
21.5
16.2%
16.4%
20.7%
24.0%
MobileCLIP2-S2
35.8
18.1%
15.4%
21.5%
22.8%
MobileCLIP-B
86.3
22.0%
19.3%
24.6%
27.3%
ViT-B-16-SigLIP2
92.9
22.0%
20.8%
23.1%
28.4%
mean
—
19.1%
—
21.5%
23.9%
bare = class name only; rich = hand-curated symptom paragraph;
grounded = LLM source-grounded symptom text. Held-out crops
are never trained on. Crop pools are nested (A ⊂B ⊂C), so
differences are attributable to the size of the unseen label space.
Note the reversal: mean hand-curated accuracy falls as classes grow
while source-grounded accuracy rises.
interval: within any single encoder the difference is between
1.3 and 2.6 standard errors of the seed-to-seed noise, which
alone would be suggestive, but four architecturally distinct
encoders spanning 11–86 M parameters reproduce the same
direction and nearly the same magnitude. The effect is real,
and it is small.
Part of that advantage depends on description length
Both arms exceed the encoder’s 77-token text context, so
the comparison above is between the leading portions of
each description. Regenerating both arms under a 50-word
ceiling, so that neither is truncated, gives a smaller and less
consistent difference: +1.26, +1.96, +1.69 and −0.38 points
on the four encoders, a mean of +1.13 with the largest encoder
reversing sign. This length-matched control covers all 51 held-
out classes rather than the paired 41 and so is not a like-for-
like replacement for the main comparison, but it is enough
to show that the sourcing advantage is not independent of
how much of each description the encoder actually reads. We
return to the token limit as a design constraint in Section 6.
4.4. Top-5 and a margin-ranked abstain gate
The gain grows against the honest baseline Chance
is a weak reference for an imbalanced label space, so we
Table 2
Top-5 and selective prediction with source-grounded descriptors.
acc@cov𝑋= accuracy when the 𝑋% most confident predictions
are kept.
Config
Classes
Model
Top-1
Top-5
AURC ↓
acc
A
MobileCLIP2-S0
17.6%
61.9%
0.6986
A
MobileCLIP-S1
28.6%
60.3%
0.5015
A
MobileCLIP2-S2
13.7%
61.2%
0.7426
A
MobileCLIP-B
26.2%
77.1%
0.6141
A
ViT-B-16-SigLIP2
26.8%
76.9%
0.6532
B
MobileCLIP2-S0
20.5%
70.0%
0.7077
B
MobileCLIP-S1
25.6%
63.7%
0.5944
B
MobileCLIP2-S2
19.4%
71.6%
0.7128
B
MobileCLIP-B
25.3%
76.0%
0.5941
B
ViT-B-16-SigLIP2
27.3%
74.7%
0.6623
C
MobileCLIP2-S0
21.6%
62.9%
0.7168
C
MobileCLIP-S1
24.0%
59.0%
0.643
C
MobileCLIP2-S2
22.8%
68.3%
0.7356
C
MobileCLIP-B
27.3%
74.6%
0.6169
C
ViT-B-16-SigLIP2
28.4%
67.7%
0.6155
Confidence is the top-1 minus top-2 similarity margin. Selective
accuracy rising as coverage tightens confirms the confidence signal
is correctly ordered.
also compare against the majority-class frequency prior.
At configuration A that prior is 14.8% against grounded’s
21.5%, a modest margin; at configuration C it falls to 4.2%
while grounded rises to 23.9%. The head therefore becomes
relatively stronger as the label space widens, which is the
opposite of the usual pattern and the reason we report
accuracy at three nested scales rather than one.
A field tool can propose a short list and abstain when
unsure, so raw top-1 is not the operative metric. Table 2 and
Fig. 6 show grounded top-5 at 60–77% on 16 classes, holding
at 59–75% on 51 classes, against a 9.8% five-item random
shortlist at that scale. Selective accuracy rises monotonically
as coverage tightens, confirming the confidence signal is
correctly ordered rather than inverted — a property that holds
for every encoder, every strategy and every scale we measured.
The grounded strategy’s abstention advantage, by contrast, is
scale-dependent: at the largest label space it holds the highest
top-5 on all four deployable encoders and the lowest area
under the risk–coverage curve on three of them, whereas
at the smallest label space it leads on neither. We therefore
recommend it as the deployment setting for large label spaces,
which is the regime the deployment case actually concerns,
and note that at 16 classes the hand-curated bank remains
competitive.
4.5. Known crops, and the seen/unseen dial
The same frozen backbone with a linear probe reaches
82.2% over 166 fine-grained classes (Table 3, Fig. 7), against
9.3% for the same encoder used zero-shot on its own seen
classes. Two patterns repeat here. Accuracy is flat across
the same 7.6-fold parameter range (≤1.5 points at every
scale), mirroring Section 4.1; and accuracy rises as the
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 4 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
Table 3
Known-crop accuracy: frozen backbone plus a linear probe, at three seen-set sizes.
Config
Crops
Classes
Images
MobileCLIP2-S0
MobileCLIP-S1
MobileCLIP2-S2
MobileCLIP-B
A
42,326
78.7%
78.3%
79.3%
79.2%
B
62,043
80.7%
79.3%
80.6%
80.6%
C
69,919
82.2%
81.1%
82.2%
82.1%
The same frozen encoders that perform unseen-crop zero-shot (Table 1). Accuracy rises with the seen label space because additional crops
contribute proportionally more training images than difficulty.
Table 4
WiSE-FT
weight
ensembling
on
MobileCLIP2-S0:
the
seen/unseen trade-off as a single dial (166 seen classes, 23,445
images; 51 unseen classes; protocol: nested-C).
𝛼
Seen ↑
Unseen zero-shot ↑
0.0 (frozen)
59.0%
21.6%
0.25
56.0%
14.8%
0.50
60.3%
15.5%
0.75
67.7%
18.4%
1.0 (naive fine-tune)
81.5%
16.4%
𝛼= 0 reproduces the frozen baseline, validating the interpolation.
𝛼= 1 is naive fine-tuning; here it still retains 8.4× chance on unseen
crops, so the dial trades cross-crop transfer for seen accuracy rather
than destroying it. 𝛼= 1.00 maximises the sum of the two columns.
seen label space grows (by roughly three points from 97 to
166 classes, for every encoder), the opposite of the unseen
side, because additional crops contribute proportionally more
training images than decision difficulty. Each head is therefore
deployed where its scaling behaviour is favourable. WiSE-FT
(Table 4, Fig. 8) turns the balance into one tunable knob,
though the knob is less forgiving than we previously reported.
Interpolation is not free at the midpoint: 𝛼= 0.5 buys only
1.3 points of seen accuracy while costing 6.2 points of unseen.
The dial pays off at its upper end, where 𝛼= 0.75 gains 8.8
points of seen for 3.2 points of unseen, and full fine-tuning
(𝛼= 1) gains 22.5 points of seen for 5.2 of unseen. Notably,
even naive fine-tuning does not destroy cross-crop transfer
here: unseen accuracy settles at 16.4%, still 8.4 times chance,
so on this task the dial trades transfer for seen accuracy rather
than erasing it. The seen column carries a few points of noise
because each 𝛼is scored with its own freshly fitted linear
head, which is why we read the trend rather than any single
interior point.
4.6. Supervised baselines and leave-one-crop-out
Fourteen supervised baselines — twelve convolutional
networks and two hybrid convolution–transformer models
— spanning 1.7–42.8 M parameters, all on the identical 166-
class seen set for 4 epochs (Table 5; per-epoch curves in
Fig. 9), are strong on known crops: the best reaches 89.3%,
about 7.1 points above the frozen probe. We report this plainly,
because for a fixed, known label set a conventional classifier
is the better choice.
Capacity is not the lever, and the complete sweep states
it plainly: the strongest of the fourteen is EfficientNet-B0
at 4.2 M with 89.3%, which is ten times smaller than the
largest model tested and beats it by 2.5 points — ResNet-
101, at 42.8 M, reaches only 86.7%. Across a 25× parameter
range the family spans 9.2 points with no monotone trend,
and the rank correlation between size and accuracy is weak
(Spearman 𝜌= 0.36). That reproduces Section 4.1 and
the seen probe above in a third, fully supervised setting,
making “size is not the bottleneck” a property of the task
rather than an artefact of frozen backbones. We do not claim
capacity is irrelevant: the smallest network, MobileNetV3-
Small at 1.7 M, is 9.2 points behind the best, so there is a
floor. The claim is narrower and better supported — past a few
million parameters, additional capacity buys little on this task.
The sharpest control is FastViT-SA12 (10.7 M), the same
architecture family as our image encoder (11.4 M) at nearly
the same size: 86.5% trained supervised against 82.2% as a
frozen probe, so the 4.3-point difference is training regime,
not architecture. Yet none of the fourteen has an output unit
for an unseen class, so cross-crop accuracy is not merely low
— it is structurally undefined at any parameter count.
Table 6 shows the two hardest crops overall are both
trained crops, so difficulty is not simply a function of whether
a crop was held out. The held-out crops do, however, average
above the trained ones on this split, and we do not claim
the held-out set is adversarially hard; the leave-one-crop-out
spread (from 4.8% to 28.9% across crops) is better read as
evidence that per-crop difficulty varies far more than the
held/trained distinction does.
4.7. On-device efficiency
Table 7 and Fig. 10 report the deployable image encoder
on a laptop processor. The 11.4 M tier runs at 17.4 ms/image
(∼58 img/s), with ONNX Runtime about 2.7× faster than
eager PyTorch, and compresses 3.5× to 12.9 MB under INT8.
4.8. INT8 is architecture-dependent, not
universally beneficial
A naive dynamic-quantisation call — the default recipe in
most tutorials — makes the lightweight tiers 18–19× slower.
A properly configured static quantise–dequantise (QDQ)
pipeline (shape-inference pre-pass, per-channel weights,
calibrated activations) recovers 5–9× of that but still does
not beat FP32 on those tiers. The pure-transformer model
behaves oppositely and becomes 2.2× faster.
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 5 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
Table 5
Supervised CNN baselines on the identical seen set.
Architecture
Params (M)
Classes
Seen top-1 ↑
Unseen
efficientnet-b0
4.2
89.3%
0 (structural)
tf-efficientnetv2-s
20.4
88.7%
0 (structural)
mobilenetv3-large-100
4.4
87.9%
0 (structural)
densenet121
7.1
87.6%
0 (structural)
resnet50
23.9
87.6%
0 (structural)
convnextv2-tiny
28.0
87.1%
0 (structural)
convnextv2-nano
15.1
87.0%
0 (structural)
regnety-040
19.7
86.9%
0 (structural)
resnet101
42.8
86.7%
0 (structural)
fastvit-sa12
10.7
86.5%
0 (structural)
mobilenetv4-conv-medium
8.7
84.9%
0 (structural)
mobilenetv4-conv-small
2.7
83.2%
0 (structural)
fastvit-t8
3.4
82.5%
0 (structural)
mobilenetv3-small-100
1.7
80.0%
0 (structural)
For a fixed, known label set a CNN is the stronger classifier.
Accuracy does not track parameter count: the strongest network is
efficientnet-b0 at 4.2 M, which is 10× smaller than the largest model
here and beats it by 2.5 points. None of them, however, has an
output unit for an unseen class, so cross-crop accuracy is not low
but undefined — the capability the descriptor head supplies.
Table 6
Leave-one-crop-out on MobileCLIP2-S0/dfndr2b (78 classes,
chance 1.3%).
Crop
𝑁
Zero-shot ↑
95% CI
Orange
1,361
28.9%
[26.4%, 31.3%]
Coffee
1,582
15.8%
[14.0%, 17.7%]
Apple
10,000
12.7%
[12.0%, 13.3%]
Peach
1,107
12.3%
[10.4%, 14.2%]
Potato
2,698
9.5%
[8.3%, 10.7%]
Corn
9,403
4.8%
[4.4%, 5.3%]
Pooled
26,151
10.5%
[10.2%, 10.9%]
Held-out crops fall in the middle of the range and two trained crops
are the hardest, so the headline held-out set is not a favourable split.
The mechanism is visible in the quantised graphs: count-
ing convolutions the quantiser could not convert gives 119,
215 and 235 for the three hybrid tiers against 3 for the pure
transformer. On the hybrids the runtime must dequantise,
run a float convolution, and requantise hundreds of times per
inference; dynamic quantisation instead maps every convolu-
tion to an integer-convolution operator that is pathologically
slow for depthwise kernels on x86.
Practical guidance For hybrid convolution–transformer
encoders on a CPU runtime, INT8 is a size lever, not a speed
lever; deploy FP32. For transformer encoders it is both. We
flag rather than assume that Arm processors, whose vector
extension provides well-optimised INT8 depthwise kernels,
may reverse this for the hybrid tiers.
5. Discussion
The contribution is a system, not a backbone The tiers
are off-the-shelf encoders used frozen; the novelty is the
source-grounded descriptor head, the margin-ranked abstain
gate, the WiSE-FT knob, and the real-time edge packaging.
Because accuracy is flat with size, the reason to offer a family
is the measured latency/size Pareto, not accuracy.
Source-grounding is a trust contribution, and the au-
thoring route that scales Grounded generation requires
one retrievable source per class, where hand-curation requires
an expert per crop and, in the retrieval form used here, cannot
even keep prototypes distinct as the label space grows. That
asymmetry is what we consider the paper’s transferable claim
beyond agriculture, and the ungrounded control in Section 4.3
settles the question it raises: the sourcing constraint does not
cost accuracy but earns a small amount of it, +1.96 points
on average and positive on all four deployable encoders.
The practical reading is not that grounding is an accuracy
technique — two points would be a thin reason to adopt
anything — but that the trade-off practitioners might expect to
face between auditability and performance does not appear in
our measurements. Where accuracy does move is descriptor
authoring as a whole: at the focused scale, replacing a class
name with a full symptom description is worth 8.8 points, four
times the sourcing effect. Authoring is the lever; sourcing is
what makes the lever safe to pull in a domain where a wrong
answer costs a harvest.
Honesty on absolutes Fine-grained cross-crop top-1 is
modest (17–29%); we lead with top-5 and the margin-ranked
abstain gate as the field-relevant metrics, and frame cross-
crop transfer as a hard open problem we advance at edge scale
rather than claim to solve.
6. Limitations
Descriptor coverage and two tiers of citation Across
18 crops, 156 of 217 disease records are LLM-filled; the
remaining 61 are explicit stubs that fall back to the hand-
curated bank. Of the 156, 112 carry a verbatim quote. We
distinguish two tiers of provenance. A quote is page-verified
when we retrieved the cited page and read the sentence there;
it is model-recalled when the generating model supplied both
quote and URL without browsing. Forty records carry at
least one page-verified field, and in 16 of those every cited
field is page-verified. Verification was prioritised by where
it matters: of the 47 filled records on held-out crops — the
set carrying this paper’s cross-crop claim — 23 are verified.
Model-recalled quotes are reported as unverified provenance
rather than as evidence, and the per-URL audit trail is released
with the data so any citation can be checked independently.
The grounding prompt over-refuses Because the gener-
ation endpoint cannot browse, a strict “never cite what you
cannot quote” instruction makes the model return empty
records rather than risk fabrication, including for well-
documented diseases. A bulk re-run converted only 1 of
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 6 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
Table 7
On-device cost of the deployable image encoder (CPU, batch 1, 224 × 224).
Model
Params (M)
Torch FP32 (ms) ↓
ONNX FP32 (ms) ↓
INT8 dyn. (ms) ↓
INT8 static (ms) ↓
FP32 (MB) ↓
MobileCLIP2-S0
11.4
47.1
17.4
320.0
62.2
45.81
MobileCLIP-S1
21.5
99.8
33.7
614.9
79.5
86.51
MobileCLIP2-S2
35.8
119.0
49.2
923.8
98.7
143.62
MobileCLIP-B
86.3
130.7
100.8
60.4
46.4
345.55
Latency in ms (median of 50 runs), ONNX Runtime 1.26.0. INT8 makes the hybrid conv–transformer tiers slower while shrinking them
∼3.5×; only the pure-transformer model gets faster.
66 stubs. This is the anti-hallucination guarantee behaving
correctly, but it means grounded coverage scales with human-
supplied sources, not with more API spend.
Label noise in the dataset Several held-out “diseases”
are not diseases: three are breeding resistance ratings, for
which no symptom descriptor can exist; one is an insect pest
rather than a pathogen; and one is a misspelling of a genus.
We recommend excluding the rating labels and report this
as a dataset-quality finding, since label noise of this kind
directly inflates the apparent class count of any fine-grained
benchmark.
Descriptor length exceeds the text context The encoders
inherit CLIP’s 77-token text window, and our generated
descriptions are longer than it: essentially every generated
prototype is truncated, and the ungrounded arm, whose
descriptions are the longer of the two, loses proportionally
more of its text. The truncated portions retain visual symptom
vocabulary in almost every case, so the comparison remains
meaningful, but it is a comparison between the leading
sentences of each description rather than between the records
as authored. A descriptor protocol written to fit the context
window — visual symptoms first, taxonomy last, or a
summarisation pass before embedding — is the obvious next
refinement, and we expect it to raise both generated arms
rather than to separate them.
Scope and variance Configuration C covers 18 of the
dataset’s crops, not all of it. Results use a single random seed
on a single evaluation machine; apart from the leave-one-crop-
out bootstrap intervals we report no variance estimates. As a
partial check, configuration C re-measured three weeks apart
on an independently rebuilt embedding cache reproduced to
within 0.15 percentage points.
Sub-10 M gap No off-the-shelf aligned model exists below
about 11 M parameters; a ∼5 M tier would require weight-
inherited distillation and is framed as future work.
7. Conclusion
A single frozen, compact, descriptor-driven VLM brings
cross-crop plant-disease diagnosis to laptops and small de-
vices at $0/image: cross-crop zero-shot at 2.2–13.9× chance,
averaged over the four deployable tiers, on 16–51 unseen
classes, 82.2% on 166 known classes (89.3% for the strongest
supervised CNN), grounded top-5 of 60–77% at 16 unseen
classes and 59–75% at 51, a margin-ranked abstain gate, and
17.4 ms real-time CPU inference at 45.8 MB, compressible to
12.9 MB under INT8 if size outweighs latency. Model size is a
near-non-factor across the 11.4–304 M range probed in Fig. 1,
and across the 11.4–86.3 M deployable family in every other
experiment. The levers are, in order: the descriptor authoring
method — and specifically the finding that source-grounded
descriptors keep improving as the unseen label space grows
— honest abstention, and the frozen-backbone hybrid.
CRediT authorship contribution statement
Pinni Venkata Abhiram: Conceptualization, Methodol-
ogy, Software, Validation, Formal analysis, Investigation,
Writing – original draft, Writing – review and editing,
Visualization, Project administration. Gaurav Shrivastava:
Methodology, Software, Validation, Investigation, Writing
– review and editing. Rahul Ananthasayanam: Validation,
Investigation, Writing – review and editing.
Declaration of competing interest
The authors declare that they have no known competing
financial interests or personal relationships that could have
appeared to influence the work reported in this paper.
Data availability
All data underlying this study are public. The images
are the SAGE crop-disease dataset [2] at revision bc9bd2899f,
released under the MIT licence and used without modification
other than the crop and per-class subsetting and the 288-pixel
resizing described in Section 3.1. The revision is quoted
because a later release of SAGE changes the class inventory;
see Section 3.1. The source-grounded descriptor registry,
every result file including per-epoch training curves, the
table and figure generators, and the full experimental code
are released at https://doi.org/PENDING-ZENODO-DOI, which
reproduces every number and figure in this article from the
released result files.
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 7 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
Funding
This research did not receive any specific grant from
funding agencies in the public, commercial, or not-for-profit
sectors.
Declaration of generative AI in the writing
process
During the preparation of this work the authors used
a large language model to generate the source-grounded
symptom descriptions that form the descriptor registry (Sec-
tion 3.1); this is a methodological component of the study
and is described in full in the Methods. The authors reviewed
and edited the manuscript text and take full responsibility for
the content of the publication.
References
[1] Arshad, M.A., Roy, T., Shen, Y., Elango, D., Chiranjeevi, S., Singh,
A.K., Ganapathysubramanian, B., Hegde, C., Singh, A., Sarkar, S.,
2026a. SAGE: Scalable agentic grounded evaluation for crop disease
diagnosis, in: CVPR 2026 Workshop on Emerging Directions in Data
for Multimodal Foundation Models. URL: https://arxiv.org/abs/
2605.09768.
[2] Arshad, M.A., Roy, T., Shen, Y., Elango, D., Chiranjeevi, S.,
Singh, A.K., Ganapathysubramanian, B., Hegde, C., Singh, A.,
Sarkar, S., 2026b.
SAGE: Scalable agentic grounded eval-
uation for crop disease diagnosis (dataset).
HuggingFace
Datasets.
URL: https://huggingface.co/datasets/tirtho149/SAGE/
tree/bc9bd2899f19379be29c7a99d37d2e89bf8e430d. [dataset]. Revision
bc9bd2899f, 7 May 2026.
[3] Bouacida, I., Farou, B., Djakhdjakha, L., Seridi, H., Kurulay, M.,
2025. Innovative deep learning approach for cross-crop plant disease
detection: A generalized method for identifying unhealthy leaves.
Information Processing in Agriculture 12, 54. doi:10.1016/j.inpa.
2024.03.002.
[4] Faghri, F., Vasu, P.K.A., Koc, C., Shankar, V., Toshev, A., Tuzel,
O., Pouransari, H., 2025.
MobileCLIP2: Improving multi-modal
reinforced training. Transactions on Machine Learning Research URL:
https://arxiv.org/abs/2508.20691.
[5] Gu, J., Stevens, S., Campolongo, E.G., Thompson, M.J., Zhang, N.,
Wu, J., et al., 2025. BioCLIP 2: Emergent properties from scaling
hierarchical contrastive learning, in: Advances in Neural Information
Processing Systems (NeurIPS). URL: https://arxiv.org/abs/2505.
23883.
[6] He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning
for image recognition, in: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 770–778.
[7] Howard, A., Sandler, M., Chu, G., Chen, L.C., Chen, B., Tan, M.,
Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q.V., Adam, H.,
2019. Searching for MobileNetV3, in: Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), pp. 1314–1324.
[8] Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N.,
Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi,
H., Farhadi, A., Schmidt, L., 2021. OpenCLIP. doi:10.5281/zenodo.
5143773.
[9] Menon, S., Vondrick, C., 2023. Visual classification via description
from large language models, in: International Conference on Learning
Representations (ICLR).
[10] Mohanty, S.P., Hughes, D.P., Salathé, M., 2016. Using deep learning
for image-based plant disease detection. Frontiers in Plant Science 7,
1419.
[11] Nguyen, K.Q., Le, L.T.T., Quach, L.D., 2025. A vision-language
foundation model for leaf disease identification. Expert Systems with
Applications doi:10.1016/j.eswa.2025.129701.
[12] Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M.,
Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.,
2024. DINOv2: Learning robust visual features without supervision.
Transactions on Machine Learning Research .
[13] Pratt, S., Covert, I., Liu, R., Farhadi, A., 2023.
What does a
platypus look like? generating customized prompts for zero-shot
image classification, in: Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), pp. 15691–15701.
[14] Qin, D., Leichner, C., Delakis, M., Fornoni, M., Luo, S., Yang, F.,
Wang, W., Banbury, C., Ye, C., Akin, B., Aggarwal, V., Zhu, T., Moro,
D., Howard, A., 2024. MobileNetV4: Universal models for the mobile
ecosystem, in: European Conference on Computer Vision (ECCV), pp.
78–96.
[15] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S.,
Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.,
2021. Learning transferable visual models from natural language su-
pervision, in: International Conference on Machine Learning (ICML),
pp. 8748–8763.
[16] Tschannen, M., Gritsenko, A., Wang, X., Naeem, M.F., Alabdulmohsin,
I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., Hénaff,
O., Harmsen, J., Steiner, A., Zhai, X., 2025. SigLIP 2: Multilingual
vision-language encoders with improved semantic understanding,
localization, and dense features. arXiv preprint arXiv:2502.14786
URL: https://arxiv.org/abs/2502.14786.
[17] Vasu, P.K.A., Pouransari, H., Faghri, F., Vemulapalli, R., Tuzel, O.,
2024. MobileCLIP: Fast image-text models through multi-modal
reinforced training, in: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 15963–15974.
[18] Wortsman, M., Ilharco, G., Kim, J.W., Li, M., Kornblith, S., Roelofs,
R., Lopes, R.G., Hajishirzi, H., Farhadi, A., Namkoong, H., Schmidt,
L., 2022. Robust fine-tuning of zero-shot models, in: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 7959–7971.
[19] Wu, K., Peng, H., Zhou, Z., Xiao, B., Liu, M., Yuan, L., Xuan, H.,
Valenzuela, M., Chen, X., Wang, X., Chao, H., Hu, H., 2023. TinyCLIP:
CLIP distillation via affinity mimicking and weight inheritance, in:
Proceedings of the IEEE/CVF International Conference on Computer
Vision (ICCV), pp. 21970–21980.
[20] Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L., 2023. Sigmoid loss
for language image pre-training, in: Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), pp. 11975–
11986.
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 8 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
Figure 1: Zero-shot accuracy against image-encoder size. Cross-
crop accuracy for frozen pretrained encoders spanning a 27-fold
parameter range, with a from-scratch small student for contrast.
The pretrained models cluster together regardless of size; the
from-scratch student sits near chance.
Figure 2: Catastrophic forgetting when specialising a frozen
encoder. Held-out zero-shot accuracy per epoch while adapting
the encoder to the seen crops. Unseen-crop accuracy falls as
the training objective improves.
Figure 3: Encoder bake-off on the pilot held-out set. Rich-
descriptor zero-shot accuracy for each candidate backbone
against its image-encoder parameter count. The biological and
leaf-disease foundation models fall at or below chance under a
descriptor protocol.
Figure 4: Mean zero-shot accuracy over the four deployable
encoders as the unseen label space grows. Source-grounded
descriptors improve as the label space widens. The keyword-
retrieved bank is shown for reference only: its prototypes collide
(Section 4.3), so the gap between the two is not a measurement
of grounding.
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 9 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
Figure 5: Descriptor strategy by encoder at each held-out scale. Per-encoder top-1 accuracy for the four descriptor strategies at
16, 34 and 51 unseen classes. The source-grounded strategy is the tallest bar for every encoder at the largest scale, whereas the
hand-curated strategy leads only at the smallest.
Figure 6: Risk–coverage behaviour of the abstain gate. Selective
accuracy as a function of the fraction of images the model
chooses to answer, using the top-1 minus top-2 similarity
margin as the confidence score. Accuracy rises monotonically
as coverage tightens, confirming the score is correctly ordered.
Figure 7: Known-crop accuracy against the size of the seen
label space. Linear-probe top-1 for four frozen encoders at 97,
154 and 166 seen classes. Accuracy rises with the label space
and is flat across a 7.6-fold parameter range.
Figure 8: The seen–unseen trade-off as a single interpolation
weight. Accuracy on seen and unseen crops as the weight-space
ensembling ratio moves from the frozen encoder to the fully
fine-tuned one. Full fine-tuning nearly halves unseen accuracy.
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 10 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
Figure 9: Training behaviour of the fourteen supervised baselines. Training loss and held-out test accuracy per epoch under one
protocol. All architectures converge, and the smallest models show the largest epoch-to-epoch variance.
Figure 10: Accuracy against on-device latency. Cross-crop zero-
shot accuracy versus single-image latency on a laptop processor;
marker area is proportional to parameter count and labels
give the 8-bit footprint. The 11.4-million-parameter tier is the
deployment sweet spot.
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 11 of 8