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