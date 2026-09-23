r"""
Phase 1 of the 2026-09-22 deep review (docs/paper/review_2026-09-22/main/review_report.md).

Creates docs/paper/revision/ as a NEW manuscript folder and applies the text-only fixes there.
docs/paper/tex/ and docs/paper/build/ are never touched, so the submitted 2026-09-22 build
stays exactly as it is.

Every edit is a GUARDED exact-string replacement: the old text must appear exactly once or the
script stops without writing anything. Numbers introduced here are either already in the
manuscript or read off paper_numbers.json; nothing new is invented.

    python scripts/paper_fixes/apply_revision_text.py            # write revision/
    python scripts/paper_fixes/apply_revision_text.py --check    # verify only, write nothing

It also prunes refs.bib to the entries the revision actually cites, so the submission package
carries no leftovers from the pre-salvage draft.

What it does NOT do, because only the authors can:
  * mint the Zenodo DOI (PENDING-ZENODO-DOI is left in place so preflight keeps failing on it);
  * say when the 32-key keyword bank was curated (a % AUTHORS: marker is inserted).
"""
from __future__ import annotations

import argparse
import re
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PAPER = REPO / "docs" / "paper"
SRC = PAPER / "tex"
DST = PAPER / "revision"

# --------------------------------------------------------------------------------------------
# (label, old, new). Order is irrelevant; each old string must occur exactly once in main.tex.
# --------------------------------------------------------------------------------------------
EDITS: list[tuple[str, str, str]] = []


def edit(label: str, old: str, new: str) -> None:
    EDITS.append((label, old, new))


# ---- title and abstract ---------------------------------------------------------------------
edit(
    "title: scope to CPU",
    r"""\title[mode=title]{Compact Vision--Language Models for Cross-Crop Plant-Disease Diagnosis at the
Edge}""",
    r"""\title[mode=title]{Compact Vision--Language Models for Cross-Crop Plant-Disease Diagnosis at the
Edge: A CPU-Only Study}""",
)

edit(
    "abstract: authoring vs encoder upgrade",
    r"""held-out sets of 16, 34 and 51 unseen classes. First, authoring outweighs model size: the best
descriptions gain 10.3--16.2 points over a bare class name, against 4.0--5.1 points between the
smallest and largest encoder, and authoring exceeds size at every configuration on both the full and
a de-duplicated label set.""",
    r"""held-out sets of 16, 34 and 51 unseen classes. First, authoring outweighs the encoder upgrade: the
best descriptions gain 10.3--16.2 points over a bare class name, against 4.0--5.1 points between the
smallest and largest encoder, and at 51 classes it leads on both label sets and under either measure
of the encoder effect.""",
)

edit(
    "abstract: per-encoder margin is a mean",
    r"""photograph reach 30.9\% top-1 on 51 unseen classes, 7.4 times the majority-class prior, and beat
source-grounded descriptions by 7.0 points on all four encoders.""",
    r"""photograph reach 30.9\% top-1 on 51 unseen classes, 7.4 times the majority-class prior, and beat
source-grounded descriptions on all four encoders, by 7.0 points on average.""",
)

edit(
    "abstract: CPU-only deployment",
    r"""The smallest encoder runs at 17.4\,ms per image on a laptop CPU in 45.8\,MB. For edge deployment,
effort spent on descriptions returns more than effort spent on parameters.""",
    r"""The smallest encoder runs at 17.4\,ms per image on a laptop CPU in 45.8\,MB. For CPU-only
deployment, effort spent on descriptions returns more than effort spent on parameters.""",
)

# ---- introduction ----------------------------------------------------------------------------
edit(
    "intro: uncited opening claim",
    r"""Vision--language models (VLMs) served from the cloud now diagnose plant disease with high accuracy
in the wild, but they are metered per query and need a network connection.""",
    r"""Vision--language models (VLMs) served from the cloud can diagnose plant disease, but they are
metered per query and need a network connection.""",
)

edit(
    "contribution 1: narrow claim, four encoders",
    r"""\item \textbf{Descriptor authoring is the dominant lever, ahead of model size.} Across seven
authoring strategies, three nested label spaces and five encoders, the best descriptions gain
10.3--16.2 points of unseen-crop accuracy over a bare class name, while moving from an 11.4\,M to an
86.3\,M encoder gains 4.0--5.1. Authoring exceeds size at every configuration, on both the full label
set and a de-duplicated one (Section~\ref{sec:lever}).""",
    r"""\item \textbf{Descriptor authoring is the dominant lever, ahead of the encoder upgrade.} Across
seven authoring strategies, three nested label spaces and four deployable encoders, the best
descriptions gain 10.3--16.2 points of unseen-crop accuracy over a bare class name, while moving
from an 11.4\,M to an 86.3\,M encoder gains 4.0--5.1. At 51 classes the lead holds on both label
sets and under either way of measuring the encoder effect (Section~\ref{sec:lever}).""",
)

edit(
    "contribution 2: label-space qualifier and added constraints",
    r"""\item \textbf{Photo-descriptive per-class sentences are the best authoring method tested.} Written
in the manner of Customized Prompts via Language models (CuPL)~\citep{pratt2023}, they reach 30.9\% on
51 unseen classes, 7.4 times the majority-class prior, and lead every other strategy at the two
larger label spaces on every deployable encoder (Section~\ref{sec:lever}).""",
    r"""\item \textbf{Photo-descriptive per-class sentences are the best authoring method tested at 34 and
51 classes.} Written in the manner of Customized Prompts via Language models
(CuPL)~\citep{pratt2023}, under three added constraints (Section~\ref{sec:desc-gen}), they reach
30.9\% on 51 unseen classes, 7.4 times the majority-class prior, and lead every other strategy at
the two larger label spaces on every deployable encoder. At 16 classes the hand-curated keyword bank
leads instead (Section~\ref{sec:lever}).""",
)

edit(
    "contribution 4: CPU benchmark, no Arm target",
    r"""\item \textbf{An edge benchmark} with measured CPU latency and INT8 footprint, and the finding that
INT8 slows hybrid convolution--transformer encoders on x86 while accelerating the pure transformer
(Section~\ref{sec:edge}).""",
    r"""\item \textbf{A CPU benchmark} with measured latency and INT8 footprint on an x86 laptop, including
the deployment observation that default INT8 quantisation slows the hybrid
convolution--transformer encoders on that runtime while accelerating the pure transformer
(Section~\ref{sec:edge}). No Arm or embedded target was measured.""",
)

# ---- related work ------------------------------------------------------------------------------
edit(
    "related work: priority claim, prompt learning, random descriptors",
    r"""object-recognition benchmarks. We implement both and compare them with source-grounded, keyword and
class-name descriptions on cross-crop plant-disease transfer, a setting in which neither has been
evaluated.""",
    r"""object-recognition benchmarks. We implement both and compare them with source-grounded, keyword and
class-name descriptions on cross-crop plant-disease transfer, a setting in which, to our knowledge,
neither has been evaluated. Two nearby results bound what such a comparison can show.
\citet{roth2023waffle} report that random-word descriptors recover much of the gain attributed to
DCLIP's generated ones, so part of any descriptor gain is ensembling rather than content; our
80-template control (Section~\ref{sec:strategies}) probes the same question from the class-name
side. A third text-side lever is to \emph{learn} the prompt context on seen classes and transfer it
to unseen ones~\citep{zhou2022coop,zhou2022cocoop}. It needs labelled seen classes, which we have,
and it is the natural comparison for a claim about authoring; we do not evaluate it here.""",
)

edit(
    "related work: TinyCLIP is cited but not used",
    r"""and MobileCLIP~\citep{vasu2024}, MobileCLIP2~\citep{mobileclip2} and TinyCLIP~\citep{wu2023} compress
it. We use their \texttt{open\_clip}~\citep{ilharco2021} releases as frozen backbones, with""",
    r"""and MobileCLIP~\citep{vasu2024}, MobileCLIP2~\citep{mobileclip2} and TinyCLIP~\citep{wu2023} compress
it. We use the MobileCLIP and MobileCLIP2 \texttt{open\_clip}~\citep{ilharco2021} releases as frozen
backbones, with""",
)

# ---- methods -------------------------------------------------------------------------------------
edit(
    "methods: majority prior at every configuration",
    r"""cap. Filenames are content hashes, so the cap acts as a deterministic random sample and the subset
is exactly rebuildable. The caps also bound class imbalance: they keep the majority-class baseline at
4.2\% on the largest held-out configuration, so a frequency prior cannot masquerade as transfer. The
resulting subset is 84{,}123 images over 217 classes and 18 crops.""",
    r"""cap. Filenames are content hashes, so the cap acts as a deterministic random sample and the subset
is exactly rebuildable. The caps bound class imbalance, but not evenly across configurations: the
majority-class baseline is 4.2\% at configuration C and 14.8\% at configuration A, where four of the
seven strategies do not beat it (Section~\ref{sec:lever}). We report it alongside uniform chance
throughout, and read the smallest configuration with that in mind. The resulting subset is
84{,}123 images over 217 classes and 18 crops.""",
)

edit(
    "methods: name the supervised baseline",
    r"""For crops seen in training, the same frozen encoder with a linear probe reaches 82.2\% over 166
fine-grained classes. A supervised convolutional network trained on the same classes reaches 89.3\%
but has no output unit for an unseen class, so its cross-crop accuracy is not low but undefined.""",
    r"""For crops seen in training, the same frozen encoder with a linear probe reaches 82.2\% over 166
fine-grained classes. The strongest of 14 supervised convolutional networks trained on the same
classes, EfficientNet-B0, reaches 89.3\% but has no output unit for an unseen class, so its
cross-crop accuracy is not low but undefined.""",
)

edit(
    "strategies: keyword-bank provenance marker",
    r"""\item[rich] The class name plus a hand-curated symptom paragraph retrieved from a 32-key keyword
bank.""",
    r"""%% AUTHORS: one sentence is still missing here -- say WHEN the 32-key bank was curated and whether
%% the held-out disease names were known at the time. The bank is the best strategy at configuration
%% A, so a reader cannot otherwise rule out that it was aimed at those diseases.
\item[rich] The class name plus a hand-curated symptom paragraph retrieved from a 32-key keyword
bank.""",
)

edit(
    "strategies: rename arms to DCLIP+ / CuPL+",
    r"""\item[DCLIP] Short visual-feature phrases per class, elicited with the question form of
\citet{menon2023}, each embedded as ``\emph{class}, which has \emph{feature}''; the class score is the
mean similarity to its features.
\item[CuPL] Twelve full sentences per class describing what the disease looks like in a photograph,
elicited with the question forms of \citet{pratt2023} and averaged into one prototype.
\end{description}""",
    r"""\item[DCLIP+] Short visual-feature phrases per class, elicited with the question form of
\citet{menon2023}, each embedded as ``\emph{class}, which has \emph{feature}''; the class score is the
mean similarity to its features.
\item[CuPL+] Twelve full sentences per class describing what the disease looks like in a photograph,
elicited with the question forms of \citet{pratt2023} and averaged into one prototype.
\end{description}

We write DCLIP+ and CuPL+, not DCLIP and CuPL, because our prompts add three constraints to the
published question forms, one of them informed by the held-out images
(Section~\ref{sec:desc-gen}). The comparison is therefore with our implementations of those
methods, not with the methods as published.""",
)

edit(
    "generation: disclose all three added constraints",
    r"""The \textbf{DCLIP} and \textbf{CuPL} registries were generated by an LLM in a chat interface, one
crop per request, from the published question forms of each method with the category filled in. We
added one instruction to both: describe the plant part the photograph actually shows. Several
held-out classes are photographed as fruit or cereal heads rather than leaves (cigar end rot, for
example, as a blackened banana tip), and a leaf description of those would describe something absent
from the image. This instruction was informed by inspecting a few images, so both baselines received
guidance the grounded registry did not; it strengthens them, which is the conservative direction for
the comparison. All 51 held-out classes were filled for both methods. Prompts, raw replies and the
resulting registries are released.""",
    r"""The \textbf{DCLIP+} and \textbf{CuPL+} registries were generated by Claude Opus 5 in a chat
interface, one crop per request, from the published question forms of each method with the category
filled in. We added three constraints to both. First, every item must name visible attributes:
colour, lesion shape, texture, margin and distribution. Second, the text must describe only what a
camera records, with no treatment advice, life cycle or pathogen taxonomy. Third, it must describe
the plant part the photograph actually shows: several held-out classes are photographed as fruit or
cereal heads rather than leaves (cigar end rot, for example, as a blackened banana tip), and a leaf
description of those would describe something absent from the image.

Two of these constraints carry information the source-grounded registry did not receive. The third
names specific held-out diseases and was written after inspecting held-out images, so it draws on
the test set; the second is itself the register whose effect Section~\ref{sec:decomp} tries to
measure. Both favour the arm that wins, so the CuPL+ margin over source-grounded text is an upper
bound on that comparison rather than a conservative estimate. All 51 held-out classes were filled
for both methods, including the four labels for which no symptom description can exist
(Section~\ref{sec:data}); the source-grounded registry filled 47 and falls through to the keyword
bank on the others. The full prompts, raw replies and resulting registries are released.""",
)

edit(
    "evaluation: describe the 2.4-point figure exactly",
    r"""shared cells are bit-identical. Each descriptor registry is a single generation, so we read
differences below the 2.4-point standard deviation measured between independently generated
registries (Section~\ref{sec:decomp}) as within noise.""",
    r"""shared cells are bit-identical. Each descriptor registry is a single generation, so we treat
differences of the order of the 2.4-point standard deviation measured between independently
generated registries (Section~\ref{sec:decomp}) as not robustly distinguishable. That figure was
measured on API-generated registries of the source-grounded schema; we assume, but did not measure,
a comparable spread for the chat-generated DCLIP+ and CuPL+ registries.""",
)

# ---- results ---------------------------------------------------------------------------------------
edit(
    "results: authoring-vs-encoder paragraph",
    r"""\paragraph{Authoring outweighs model size} Replacing a bare class name with the best descriptions
gains 16.2, 10.3 and 11.8 points at configurations A, B and C. Moving from the 11.4\,M to the
86.3\,M encoder under the best descriptions gains 5.1, 4.0 and 4.7. Authoring therefore exceeds size
by 2.5--3.1 times on the full label set. On the de-duplicated set the ratio is 1.2--2.8, the low end
at configuration B, where the encoder spread widens to 10.0 points; authoring still exceeds size at
every configuration on both label sets. The size gain is also the costlier of the two: it takes
5.8 times the on-device latency and 7.5 times the footprint (Section~\ref{sec:edge}), whereas better
descriptions are authored once, offline, and cost nothing per image.""",
    r"""\paragraph{Authoring outweighs the encoder upgrade} Replacing a bare class name with the best
descriptions gains 16.2, 10.3 and 11.8 points at configurations A, B and C. Moving from the
11.4\,M to the 86.3\,M encoder under those same descriptions gains 5.1, 4.0 and 4.7, so authoring
exceeds the encoder upgrade by 2.5--3.1 times on the full label set. Three qualifications bound that
statement. First, the best strategy is chosen per configuration after the fact, and the encoder term
is an endpoint difference rather than the spread across the four encoders;
Table~\ref{tab:authoring} reports both. Second, on the de-duplicated set the ratio falls to
1.2--2.8, and at configuration B the two gains differ by 2.1 points, inside the 2.4-point registry
spread of Section~\ref{sec:eval}, so there they are not distinguishable. Third, accuracy is not
monotone in encoder size: the 35.8\,M encoder is the weakest of the four at configuration A under
six of the seven strategies, and the two endpoints differ in pretraining data as well as in size, so
the second lever is better read as the choice of encoder than as its parameter count. At 51 classes,
the largest label space, the authoring gain exceeds the encoder term on both label sets and under
either measure. The encoder term is also the costlier of the two: it takes 5.8 times the latency and
7.5 times the footprint (Section~\ref{sec:edge}), whereas better descriptions are authored once,
offline, and cost nothing per image.""",
)

edit(
    "results: which method wins, CuPL+ naming and noise wording",
    r"""methods cannot collide. CuPL leads at B and C with 30.7\% and 30.9\%, ahead of source-grounded text by
8.0 and 7.0 points and of the keyword bank by 5.7 and 9.4. Figure~\ref{fig:perencoder} shows the
lead is not a property of one backbone: at B and C, CuPL beats source-grounded text on all four
deployable encoders, by 4.8--10.1 points at B and 5.5--8.6 at C, and on the de-duplicated set it
leads on all four encoders at every configuration. The margin is roughly three times the 2.4-point
spread between independently generated registries, so a single generation does not account for it.
DCLIP, by contrast, is level with source-grounded text at B and C ($+0.5$ and $+1.2$ points) and falls
7.0 points behind it at A.""",
    r"""methods cannot collide. CuPL+ leads at B and C with 30.7\% and 30.9\%, ahead of source-grounded text
by 8.0 and 7.0 points and of the keyword bank by 5.7 and 9.4. Figure~\ref{fig:perencoder} shows the
lead is not a property of one backbone: at B and C, CuPL+ beats source-grounded text on all four
deployable encoders, by 4.8--10.1 points at B and 5.5--8.6 at C, and on the de-duplicated set it
leads on all four encoders at every configuration. The margin is roughly three times the 2.4-point
spread between independently generated registries, so it is unlikely to be an artefact of one
generation. DCLIP+, by contrast, is level with source-grounded text at B and C ($+0.5$ and $+1.2$
points) and falls 7.0 points behind it at A.""",
)

edit(
    "results: decomposition, DCLIP+ as the register control",
    r"""\paragraph{Text} What remains, 6.7 points at configuration C on the full label set and 5.9 on the
de-duplicated set, lies in the text. The source-grounded registry follows the SAGE schema of pathogen,
affected organs and symptoms, written in the register of a pathology reference; CuPL's sentences are
written as descriptions of a photograph. CLIP's text tower was trained on image captions, and the
second register is closer to them. We name this as the likely explanation rather than a demonstrated
one: the CuPL registry also differs from the grounded one in the model and interface that generated
it, and a registry that is both source-grounded and photo-descriptive, which would separate register
from provenance, is the natural next experiment.""",
    r"""\paragraph{Text} What remains, 6.7 points at configuration C on the full label set and 5.9 on the
de-duplicated set, lies in the text. The source-grounded registry follows the SAGE schema of pathogen,
affected organs and symptoms, written in the register of a pathology reference; CuPL+'s sentences are
written as descriptions of a photograph. CLIP's text tower was trained on image captions, and the
second register is closer to them. We offer this as a plausible contributor, not a demonstrated
cause. The CuPL+ registry differs from the grounded one in the model and interface that generated
it, in sentence form, and in the three constraints of Section~\ref{sec:desc-gen}. DCLIP+ is the
relevant control and it cuts against register alone: it received the same camera-only constraint and
the same organ rule, yet it only draws level with source-grounded text at B and C and trails CuPL+
by 5.7--8.0 points, a gap as large as the one register is being asked to explain. A registry that is
both source-grounded and photo-descriptive, and one with the taxonomy fields stripped from the
existing grounded text, are the two experiments that would separate these factors.""",
)

edit(
    "results: citation control, describe the SD exactly",
    r"""accuracy. The seed-to-seed standard deviation, 2.4 points, is the registry-to-registry spread used
throughout this article. We do not claim equivalence, which would need a margin fixed in advance.""",
    r"""accuracy. The seed-to-seed standard deviation of that paired difference, 2.4 points, is the
registry-to-registry scale used throughout this article; it is an empirical scale, not a bound. We
do not claim equivalence, which would need a margin fixed in advance.""",
)

edit(
    "results: abstention paragraph, CuPL+ naming",
    r"""for declining one image in five; its value is that the system can decline rather than guess. Top-5
and abstention were not measured for CuPL, whose higher top-1 suggests but does not establish a
higher top-5.""",
    r"""for declining one image in five; its value is that the system can decline rather than guess. Top-5
and abstention were not measured for CuPL+, whose higher top-1 suggests but does not establish a
higher top-5.""",
)

edit(
    "results: CuPL+ in the pareto sentence",
    r"""times faster than eager PyTorch. Figure~\ref{fig:pareto} places the encoders on the accuracy--latency
plane under CuPL descriptions.""",
    r"""times faster than eager PyTorch. Figure~\ref{fig:pareto} places the encoders on the accuracy--latency
plane under CuPL+ descriptions.""",
)

edit(
    "results: INT8 static speed-up wording",
    r"""convolution--transformer encoders 18--19 times \emph{slower}. A static quantise--dequantise pipeline
with per-channel weights and calibrated activations recovers 5.1--9.4 times of that loss but still
does not beat FP32 on those encoders.""",
    r"""convolution--transformer encoders 18--19 times \emph{slower}. A static quantise--dequantise pipeline
with per-channel weights and calibrated activations is 5.1--9.4 times faster than the dynamic graph
but still does not beat FP32 on those encoders.""",
)

edit(
    "results: INT8 mechanism, separate the two paths",
    r"""times faster. The quantised graphs show why: the quantiser leaves 119, 215 and 235 convolutions
unconverted in the three hybrids against 3 in the transformer, so on the hybrids the runtime must
dequantise, run a floating-point convolution and requantise hundreds of times per inference. For
hybrid encoders on an x86 CPU runtime, INT8 is therefore a size lever, shrinking the smallest encoder
3.5 times to 12.9\,MB, not a speed lever.""",
    r"""times faster. The two paths fail differently. In the \emph{static} graphs the quantiser leaves 119,
215 and 235 convolutions unconverted in the three hybrids against 3 in the transformer, so the
runtime must dequantise, run a floating-point convolution and requantise hundreds of times per
inference; these are counts in the serialised graph, and the runtime fuses quantisation pairs at
session start, so they are evidence rather than proof. The \emph{dynamic} graphs leave no
floating-point convolution at all, so the larger slowdown on that path has a different cause, which
we do not isolate. For hybrid encoders on this x86 runtime, INT8 is therefore a size lever,
shrinking the smallest encoder 3.5 times to 12.9\,MB, not a speed lever; we did not measure accuracy
under INT8.""",
)

# ---- discussion ---------------------------------------------------------------------------------------
edit(
    "discussion: quote both label sets",
    r"""whether cross-crop diagnosis works. Across a 7.6-fold range of encoder size, the best descriptions
buy 2.5--3.1 times more accuracy than the largest encoder does, and they are paid for once, offline,
whereas encoder size is paid for on every image in latency and memory.""",
    r"""whether cross-crop diagnosis works. Across the four encoders tested, the best descriptions buy
2.5--3.1 times more accuracy on the full label set, and 1.2--2.8 on the de-duplicated one, than the
step from the smallest encoder to the largest does, and they are paid for once, offline, whereas the
encoder is paid for on every image in latency and memory.""",
)

edit(
    "discussion: drop the untested authoring prescription",
    r"""\paragraph{Write for the camera} Among the methods tested, descriptions written as accounts of a
photograph worked best, and the decomposition of Section~\ref{sec:decomp} locates the advantage in
the text rather than in how it is embedded or in whether it cites sources. Source-grounded text
written in the register of a pathology reference, with its pathogen and taxonomy fields, lags by about
seven points. The implication for authoring agricultural descriptors is concrete: describe colour,
shape, texture, margin and distribution as a camera would record them, and leave taxonomy out of the
prototype.""",
    r"""\paragraph{Write for the camera} At 34 and 51 classes, descriptions written as accounts of a
photograph worked best, and the decomposition of Section~\ref{sec:decomp} locates the advantage in
the text rather than in how it is embedded or in whether it cites sources. Source-grounded text
written in the register of a pathology reference, with its pathogen and taxonomy fields, lags by
about seven points. What this licenses is narrower than a rule for authoring: the winning registry
was also written under constraints the source-grounded one did not receive, and DCLIP+ shares its
camera-only constraint without sharing its lead, so which property of the text carries the advantage
is not yet isolated.""",
)

edit(
    "discussion: citation requirement is accuracy-neutral",
    r"""\paragraph{Auditability is free} Requiring every description to cite a retrievable source did not
measurably cost accuracy. A deployment that must justify its outputs, as an agricultural advisory
service may, can therefore demand source-grounded text without an accuracy penalty from the citation
requirement itself. What it should not assume is that sourcing improves accuracy; it does not, and the
register in which the grounded text is written currently costs it seven points. A registry that is both
cited and photo-descriptive would, if the register explanation holds, combine the audit trail with the
best accuracy.""",
    r"""\paragraph{The citation requirement is accuracy-neutral} Requiring every description to cite a
retrievable source did not measurably cost accuracy. A deployment that must justify its outputs, as
an agricultural advisory service may, can therefore demand source-grounded text without an accuracy
penalty from the citation requirement itself. What it should not assume is that sourcing improves
accuracy; it does not, and the source-grounded registry trails CuPL+ by seven points for reasons
Section~\ref{sec:decomp} places in the text without isolating. Nor is a citation an audit on its
own: 23 of our 47 filled held-out records carry a page-verified field and the rest carry
model-recalled citations, so the trail is only as good as the verification behind it. A registry
that is both cited and photo-descriptive would, if the register explanation holds, combine the audit
trail with the best accuracy.""",
)

edit(
    "discussion: fine-tuning is one short run",
    r"""\paragraph{What does not work} Two alternatives to a frozen compact encoder with good descriptions
are ruled out. A supervised convolutional network is the stronger choice for a fixed set of known crops,
reaching 89.3\% on 166 seen classes against 82.2\% for the frozen probe, but it cannot represent an
unseen crop at all. Fine-tuning the encoder on the seen crops raises seen accuracy but lowers unseen
accuracy, from 21.6\% to 16.4\% on the smallest encoder at configuration C, so specialisation trades
away the transfer this system exists to provide; we keep the encoder frozen.""",
    r"""\paragraph{What does not work} Neither alternative to a frozen compact encoder with good
descriptions helped here. A supervised convolutional network is the stronger choice for a fixed set
of known crops, reaching 89.3\% on 166 seen classes against 82.2\% for the frozen probe, but it
cannot represent an unseen crop at all. In one short fine-tuning run on the seen crops, on the
smallest encoder at configuration C and under the source-grounded registry, unseen accuracy fell
from 21.6\% to 16.4\%, while seen accuracy rose within that run's own protocol, which uses fewer
seen images than the probe above and is not comparable with it. A single run does not settle the
question; we keep the encoder frozen because transfer is what the system exists to provide.""",
)

# ---- limitations ------------------------------------------------------------------------------------
edit(
    "limitations: the 2.4 figure is a scale, not a bound",
    r"""\paragraph{One generation per registry} Each descriptor registry is a single generation. The
2.4-point registry-to-registry spread bounds how far any single cell could move, and the central
margins here, CuPL over source-grounded text at 7.0 points and authoring over size by 2.5--3.1 times,
are well outside it. Smaller differences, such as DCLIP against source-grounded text, are not.
Regenerating the DCLIP and CuPL registries over further seeds would turn their directions into
measured effects.""",
    r"""\paragraph{One generation per registry} Each descriptor registry is a single generation. The
2.4-point registry-to-registry standard deviation gives an empirical scale for that variability
rather than a bound, and the central margins here, CuPL+ over source-grounded text at 7.0 points and
authoring over the encoder upgrade by 2.5--3.1 times, are several times it. Smaller differences, such
as DCLIP+ against source-grounded text, are not. The scale itself was measured on API-generated
registries; regenerating the DCLIP+ and CuPL+ registries over further seeds would both measure their
own spread and turn their directions into measured effects.""",
)

edit(
    "limitations: register is one of several differences",
    r"""\paragraph{Register is confounded with provenance} The source-grounded and CuPL registries differ in
register and also in the model and interface that produced them, and the CuPL registry was generated
one crop per request, so the model saw sibling classes. We attribute the gap to the text, which the
decomposition supports, but name register as its likely cause rather than a demonstrated one.""",
    r"""\paragraph{Register is confounded with provenance} The source-grounded and CuPL+ registries differ in
register and also in the model and interface that produced them, Claude Sonnet 5 through an API
against Claude Opus 5 in a chat, and the CuPL+ registry was generated one crop per request, so the
model saw sibling classes. We attribute the gap to the text, which the
decomposition supports, but register is one of several differences between the two registries and is
not isolated by any experiment reported here.""",
)

edit(
    "limitations: the added constraints favour the winner",
    r"""\paragraph{The baselines received image-informed guidance} The instruction to describe the photographed
plant part was informed by inspecting a few images. It favours DCLIP and CuPL, the conservative
direction for a comparison that CuPL wins, but the source-grounded registry did not receive it.""",
    r"""\paragraph{The baselines received image-informed guidance} Two of the three constraints added to the
DCLIP+ and CuPL+ prompts (Section~\ref{sec:desc-gen}) were withheld from the source-grounded
registry, and one of them was written after inspecting held-out images. They favour the arm that
wins, so the CuPL+ margin over source-grounded text should be read as an upper bound. Re-scoring the
same comparison on the classes the organ rule does not touch would bound how much of the margin it
buys.""",
)

edit(
    "limitations: CuPL+ in the abstention paragraph",
    r"""\paragraph{Shortlist and abstention were measured on two strategies} Top-5 and selective prediction
are reported for the source-grounded registry and the keyword bank only, not for CuPL.""",
    r"""\paragraph{Shortlist and abstention were measured on two strategies} Top-5 and selective prediction
are reported for the source-grounded registry and the keyword bank only, not for CuPL+, although the
deployment we recommend uses CuPL+.""",
)

edit(
    "limitations: scope covers hardware and pretraining",
    r"""\paragraph{Scope} Results cover 18 crops of one dataset at a pinned revision, on one evaluation
machine. Source verification is partial: 23 of the 47 filled held-out grounded records carry a
page-verified field, and the remainder carry model-recalled citations, released for independent
checking.""",
    r"""\paragraph{Scope} Results cover 18 crops of one dataset at a pinned revision, on one evaluation
machine. Every latency here is from an x86 laptop CPU; no Arm or embedded target was measured, and
the INT8 result in particular may not transfer to one. The held-out crops are unseen by our
training, not necessarily by the encoders' web-scale pretraining, which we did not audit. Source
verification is partial: 23 of the 47 filled held-out grounded records carry a page-verified field,
and the remainder carry model-recalled citations, released for independent checking.""",
)

# ---- conclusion ----------------------------------------------------------------------------------------
edit(
    "conclusion: shortlist, label-space qualifier, CPU-only",
    r"""A frozen compact vision--language model can diagnose plant diseases on crops it was never trained on,
at 17.4\,ms per image on a laptop CPU, and how its descriptions are written decides how well. Across
seven authoring strategies, three label spaces and four encoders, the best descriptions buy more
accuracy than the largest encoder, at no per-image cost. Photo-descriptive per-class sentences work
best, reaching 30.9\% on 51 unseen classes, 7.4 times the majority-class prior. Requiring
descriptions to cite sources costs nothing measurable, and the remaining gap between cited and
photo-descriptive text lies in the text rather than in citation or embedding. The lever for edge
deployment is the descriptions, not the parameters.""",
    r"""A frozen compact vision--language model can shortlist plant diseases on crops it was never trained
on, at 17.4\,ms per image on a laptop CPU, and how its descriptions are written decides how well.
Across seven authoring strategies, three label spaces and four encoders, the best descriptions buy
more accuracy than the step to the largest encoder, at no per-image cost. At 34 and 51 classes,
photo-descriptive per-class sentences work best, reaching 30.9\% on 51 unseen classes, 7.4 times the
majority-class prior. Requiring descriptions to cite sources costs nothing measurable, and the
remaining gap between cited and photo-descriptive text lies in the text rather than in citation or
embedding. The lever for CPU-only deployment is the descriptions, not the parameters.""",
)

edit(
    "generation: correct the source-grounded registry's model",
    r"""The \textbf{grounded} registry was generated by Claude Sonnet 4.6 at temperature 0 under a strict
grounding prompt that forbids uncited claims.""",
    r"""The \textbf{grounded} registry was generated by Claude Sonnet 5 at temperature 0 under a strict
grounding prompt that forbids uncited claims.""",
)

# ---- remaining arm mentions: captions, decomposition prose, back matter --------------------------
# Mentions of the PUBLISHED methods (Related work, and "in the manner of CuPL") keep their plain
# names; only our own arms become CuPL+ / DCLIP+.
for _label, _old, _new in [
    ("figure 1 caption",
     r"""collisions grow; the per-class LLM methods do not collide, and CuPL leads at 34 and 51 classes. The
dotted line is uniform chance.}""",
     r"""collisions grow; the per-class LLM methods do not collide, and CuPL+ leads at 34 and 51 classes.
The dotted line is uniform chance.}"""),
    ("figure 2 caption",
     r"""set). At 34 and 51 classes CuPL is the tallest bar on every deployable encoder.}""",
     r"""set). At 34 and 51 classes CuPL+ is the tallest bar on every deployable encoder.}"""),
    ("decomposition opening",
     r"""Source-grounded text differs from CuPL in three respects that could each explain the gap: it is""",
     r"""Source-grounded text differs from CuPL+ in three respects that could each explain the gap: it is"""),
    ("construction paragraph",
     r"""description never reaches the encoder. CuPL embeds twelve short sentences, none truncated. To test""",
     r"""description never reaches the encoder. CuPL+ embeds twelve short sentences, none truncated. To test"""),
    ("construction result",
     r"""configuration C accuracy moves by $+0.3$ points, recovering 4\% of the gap to CuPL; on the""",
     r"""configuration C accuracy moves by $+0.3$ points, recovering 4\% of the gap to CuPL+; on the"""),
    ("figure 4 caption",
     r"""\caption{Unseen-crop top-1 under CuPL descriptions at 51 classes against single-image FP32 latency on""",
     r"""\caption{Unseen-crop top-1 under CuPL+ descriptions at 51 classes against single-image FP32 latency on"""),
    ("data availability",
     r"""DCLIP and CuPL prompts and raw replies, every result file, the table and figure generators, and the""",
     r"""DCLIP+ and CuPL+ prompts and raw replies, every result file, the table and figure generators, and the"""),
    ("AI declaration: rename the arms and point to the named models",
     r"""(Section~\ref{sec:desc-gen}) and the DCLIP and CuPL registries through a chat interface. The generated""",
     r"""and the DCLIP+ and CuPL+ registries through a chat interface; Section~\ref{sec:desc-gen} names
the model behind each. The generated"""),
]:
    edit(_label, _old, _new)


# --------------------------------------------------------------------------------------------
# Three real references the revision now cites. Metadata is deliberately minimal: no DOI or URL
# is asserted here. Run the repo's citation-verification pass before submitting.
# --------------------------------------------------------------------------------------------
NEW_BIB = r"""
@inproceedings{zhou2022coop,
  author    = {Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  title     = {Learning to Prompt for Vision-Language Models},
  journal   = {International Journal of Computer Vision},
  booktitle = {International Journal of Computer Vision},
  volume    = {130},
  number    = {9},
  pages     = {2337--2348},
  year      = {2022},
  note      = {AUTHORS: verify metadata and add DOI before submission}
}

@inproceedings{zhou2022cocoop,
  author    = {Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  title     = {Conditional Prompt Learning for Vision-Language Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
  note      = {AUTHORS: verify metadata and add DOI before submission}
}

@inproceedings{roth2023waffle,
  author    = {Roth, Karsten and Kim, Jae Myung and Koepke, A. Sophia and Vinyals, Oriol and Schmid, Cordelia and Akata, Zeynep},
  title     = {Waffling Around for Performance: Visual Classification with Random Words and Broad Concepts},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
  note      = {AUTHORS: verify metadata and add DOI before submission}
}
"""


def _prune_bib(bib: str, tex: str) -> str:
    """Keep only the entries the manuscript cites.

    docs/paper/tex/refs.bib still carries the CNN-baseline and WiSE-FT references the
    pre-salvage draft used. BibTeX ignores them, but the submission package should not ship a
    bibliography a third of which the paper never mentions, and scripts/check_tex_refs.py
    reports them as a failure. Every cited key must be present or this aborts: dropping a key
    the paper needs would be far worse than keeping a spare one.
    """
    cited = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}", tex):
        cited.update(k.strip() for k in m.group(1).split(",") if k.strip())

    entries, order = {}, []
    for block in re.split(r"(?m)^(?=@)", bib):
        m = re.match(r"@\w+\s*\{\s*([^,\s]+)\s*,", block)
        if m:
            entries[m.group(1)] = block.rstrip()
            order.append(m.group(1))

    missing = sorted(cited - set(entries))
    if missing:
        raise SystemExit(f"ABORT: cited but absent from refs.bib: {missing}")
    dropped = [k for k in order if k not in cited]
    kept = [entries[k] for k in order if k in cited]
    print(f"[patch] refs.bib: kept {len(kept)} cited entries, dropped {len(dropped)} uncited "
          f"({', '.join(dropped) if dropped else 'none'})")
    return "\n\n".join(kept) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify every anchor, write nothing")
    args = ap.parse_args()

    src_tex = SRC / "main.tex"
    text = src_tex.read_text(encoding="utf-8")

    # ---- guard: every anchor must appear exactly once -------------------------------------
    problems = []
    for label, old, _new in EDITS:
        n = text.count(old)
        if n != 1:
            problems.append(f"  [{n} matches] {label}")
    if problems:
        print(f"ABORT: {len(problems)} anchor(s) did not match exactly once in {src_tex}:")
        print("\n".join(problems))
        return 1
    print(f"[patch] {len(EDITS)} anchors matched exactly once in {src_tex.name}")

    # ---- sanity: the numbers this patch introduces come from paper_numbers.json -----------
    N = json.loads((PAPER / "paper_numbers.json").read_text(encoding="utf-8"))
    maj_a, maj_c = N["majority_prior"]["A"], N["majority_prior"]["C"]
    means_a = N["configs"]["A"]["uncleaned"]["means"]
    below = [k for k, v in means_a.items() if k != "grounded_split" and v < maj_a]
    assert abs(maj_a - 14.8) < 0.05 and abs(maj_c - 4.2) < 0.05, "majority priors moved"
    assert len(below) == 4, f"expected 4 strategies below the A prior, found {len(below)}: {below}"
    pe_a = N["configs"]["A"]["uncleaned"]["per_encoder"]
    worst_s2 = [s for s in means_a
                if s != "grounded_split"
                and min(pe_a, key=lambda m: pe_a[m][s]) == "MobileCLIP2-S2"]
    assert len(worst_s2) == 6, f"expected S2 weakest under 6 strategies, found {len(worst_s2)}"
    print(f"[patch] verified: A prior {maj_a}% (4 strategies below), C prior {maj_c}%, "
          f"S2 weakest under {len(worst_s2)} of 7 strategies")

    if args.check:
        print("[patch] --check only, nothing written")
        return 0

    # ---- build docs/paper/revision -------------------------------------------------------------
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "figures").mkdir(exist_ok=True)

    # NOTE: do not rewrap these replacements. apply_revision_results.py anchors on text this script
    # inserts, so changing the line breaks here silently breaks every Phase-2 anchor. A dozen
    # over-long source lines are the price; main.tex is generated, and the header says to edit
    # this script rather than the file.
    for label, old, new in EDITS:
        text = text.replace(old, new, 1)

    # the revision keeps its own figures next to the manuscript so the 2026-09-22 figures are frozen.
    text = text.replace(r"\graphicspath{{../figures/}}", r"\graphicspath{{figures/}}", 1)
    header = ("%% REVISION 2 (2026-09-23): the 2026-09-22 deep review applied.\n"
              "%% Generated by scripts/paper_fixes/apply_revision_text.py from ../tex/main.tex.\n"
              "%% Do not hand-edit BOTH files: ../tex/main.tex is the frozen submitted version.\n")
    text = header + text

    out_tex = DST / "main.tex"
    with open(out_tex, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"[patch] wrote {out_tex}")

    # Highlights track the revised contributions: the authoring claim is stated as a gain rather
    # than as a universal, and the decomposition line carries the taxonomy ablation. Elsevier caps
    # each one at 85 characters, so the length is asserted rather than trusted.
    highlights = [
        "Descriptions gain 10.3-16.2 points; the 11.4M-to-86.3M encoder step gains 4.0-5.1.",
        "Photo-descriptive text reaches 30.9% on 51 unseen classes, 7.4x the majority prior.",
        "Citing a source costs no accuracy; dropping taxonomy recovers none of the deficit.",
        "The smallest encoder runs at 17.4 ms per image on a laptop CPU in 45.8 MB.",
        "INT8 is a size lever, not a speed lever, for hybrid conv-transformer encoders on x86.",
    ]
    over = [h for h in highlights if len(h) > 85]
    assert not over, f"highlights over 85 characters: {[(len(h), h) for h in over]}"
    with open(DST / "highlights.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(highlights) + "\n")
    print(f"[patch] wrote {DST / 'highlights.txt'} "
          f"({len(highlights)} highlights, longest {max(len(h) for h in highlights)} chars)")

    bib = (SRC / "refs.bib").read_text(encoding="utf-8").rstrip() + "\n" + NEW_BIB
    bib = _prune_bib(bib, text)
    with open(DST / "refs.bib", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(bib)
    print(f"[patch] wrote {DST / 'refs.bib'} (+3 entries: coop, cocoop, waffle)")

    for extra in ("cas-dc.cls", "cas-common.sty", "cas-model2-names.bst"):
        p = SRC / extra
        if p.exists():
            shutil.copy2(p, DST / extra)

    print("\n[patch] next:")
    print("  python docs/paper/make_tex_tables_revision.py")
    print("  python docs/paper/make_figures_revision.py")
    print("  cd docs/paper/revision && latexmk -pdf main.tex")
    print("\n[patch] still needs an author, not a script:")
    print("  * mint the Zenodo DOI (PENDING-ZENODO-DOI is still in the manuscript)")
    print("  * name the chat model/version in Section 3.4 (% AUTHORS: marker)")
    print("  * say when the keyword bank was curated (% AUTHORS: marker)")
    print("  * verify the 3 new bib entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
