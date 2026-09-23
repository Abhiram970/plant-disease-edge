#!/usr/bin/env python3
"""Apply the 2026-09-12 audit fixes to the manuscript source.

This round differs from 2026-09-10 and 2026-09-11 in intent. Those rounds fixed issues by
ADDING hedges. Ten audits later the accretion is itself the problem: each new caveat is a
new claim, and several of the caveats now contradict each other or overreach beyond the
evidence cited for them. So most edits here DELETE or NARROW existing prose rather than
append to it, and where two sections disagree, one of them is made authoritative and the
other made to defer to it.

Every edit is a whitespace-insensitive exact match guarded by a uniqueness assertion: a
stale or ambiguous anchor aborts the run before anything is written, so a half-applied
patch is impossible. Re-running after success is a no-op.

    python scripts/paper_fixes/patch_audit_2026-09-12.py --dry-run
    python scripts/paper_fixes/patch_audit_2026-09-12.py

What this round does NOT fix, because prose cannot:
  * The Zenodo DOI is still the literal placeholder PENDING-ZENODO-DOI. Only the authors
    can mint it. preflight.py catches it.
  * No DCLIP/CuPL baseline and no 80-template bare-class ensemble. Both are runs.
  * Tables 2-6 remain single-seed. That is a run, not a sentence.
Those three are reported by the audit and left visible in Section 6 where they belong.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs" / "paper"
MAIN = PAPER / "tex" / "main.tex"
PILOT = PAPER / "tex" / "tab_pilot.tex"          # hand-maintained, no generator
GEN = PAPER / "make_tex_tables.py"               # emits tab_loco / tab_supervised captions


class Edit:
    """One whitespace-insensitive replacement with a justification."""

    def __init__(self, ident: str, path: Path, old: str, new: str, why: str) -> None:
        self.ident = ident
        self.path = path
        self.old = " ".join(old.split())
        self.new = " ".join(new.split())
        self.why = why

    def pattern(self) -> re.Pattern[str]:
        return re.compile(r"\s+".join(re.escape(w) for w in self.old.split()))

    def apply(self, text: str) -> tuple[str, str]:
        """Return (new_text, status) where status is applied | already | MISSING | AMBIGUOUS."""
        if self.pattern().search(text) is None:
            # already applied?
            if re.search(r"\s+".join(re.escape(w) for w in self.new.split()), text):
                return text, "already"
            return text, "MISSING"
        hits = len(self.pattern().findall(text))
        if hits > 1:
            return text, f"AMBIGUOUS ({hits} matches)"
        return self.pattern().sub(lambda _: self.new, text, count=1), "applied"


EDITS: list[Edit] = []


def edit(ident, path, old, new, why):
    EDITS.append(Edit(ident, path, old, new, why))


# ---------------------------------------------------------------------------
# 1. THE THESIS. Table 2's own mean row says encoder choice buys more than
#    descriptor authoring at B and C. Both numbers were already in the paper,
#    in different subsections, and were never compared.
#      bare -> grounded : +8.85 (A)  +2.35 (B)  +4.80 (C)
#      11.4M -> 86.3M   : +8.60 (A)  +4.80 (B)  +5.70 (C)
# ---------------------------------------------------------------------------
edit(
    "discussion-authoring-lever", MAIN,
    r"""Where accuracy \emph{does} move is descriptor authoring as a whole: at the focused scale,
    replacing a class name with a full symptom description is worth 8.8 points, four times the
    sourcing effect. Authoring is the lever; sourcing is what makes the lever safe to pull in a
    domain where a wrong answer costs a harvest.""",
    r"""Where accuracy \emph{does} move is descriptor authoring as a whole, though only at the
    smallest label space is it the largest lever available. Replacing a class name with a full
    symptom description is worth 8.8 points at configuration~A, four times the sourcing effect, but
    2.4 and 4.8 points at B and C; over the same configurations, moving from the 11.4\,M tier to the
    86.3\,M one under source-grounded text is worth 8.6, 4.8 and 5.7 points. Authoring therefore
    dominates encoder choice at A, ties it at B and trails it at C, and the honest ordering is that
    the two are comparable levers across this range rather than that one of them is \emph{the}
    lever. What distinguishes them is cost: the authoring gain is paid once, offline, in generation
    tokens, whereas the 5.7 points at C cost 5.8$\times$ the on-device latency and 7.5$\times$ the
    footprint (Table~\ref{tab:edge}). That asymmetry, not the point difference, is the reason to
    spend effort on descriptors. Sourcing is what makes either lever safe to pull in a domain where
    a wrong answer costs a harvest.""",
    "Table 2 contradicts 'Authoring is the lever' at B and C. Both series were already in the "
    "paper (Sec 4.1 reports the 5.7-point spread; the authoring gains follow from the mean row) "
    "but never compared. Reframed to the defensible cost argument.",
)

edit(
    "conclusion-size-null", MAIN,
    r"""Across the 11.4--86.3\,M deployable family, size buys at most a few points and
    never enough to justify what it costs in latency and footprint; on the wider 11.4--321.8\,M pilot
    survey it does not track accuracy at all. The levers are, in order: the descriptor authoring
    method, where source-grounded text is worth 8.8 points over a bare class name at the focused scale
    and is the one strategy that does not decay as the label space triples, though a de-duplicated
    re-run shows it does not improve either; then honest abstention; then the frozen-backbone hybrid.""",
    r"""Across the 11.4--86.3\,M deployable family, size buys 4.8 to 8.6 points of unseen accuracy,
    which is real but costs up to 5.8$\times$ the latency and 7.5$\times$ the footprint to obtain; on
    the wider 11.4--321.8\,M pilot survey we detect no association between capacity and accuracy at
    all, though with ten encoders that survey could not resolve one. The levers are therefore
    comparable in size and differ in price: descriptor authoring, worth 8.8 points over a bare class
    name at the focused scale and 2.4 to 4.8 points at the wider ones, is paid once and offline, and
    source-grounded text is the one strategy that does not decay as the label space triples, though a
    de-duplicated re-run shows it does not improve either; then honest abstention; then the
    frozen-backbone hybrid.""",
    "Deletes the flat null ('does not track accuracy at all') asserted from n=10, and the "
    "'at most a few points' magnitude claim that Table 2 contradicts. Keeps the cost argument.",
)

edit(
    "contrib1-size-null", MAIN,
    r"""Within the 11.4--86.3\,M deployable range the size effect is small relative to the latency
    and memory it costs, and on the 17-class pilot survey (Protocol~P) accuracy does not track
    parameter count over a 28-fold range (Sections~\ref{sec:flat}--\ref{sec:desc}).""",
    r"""Within the 11.4--86.3\,M deployable range the size effect is small relative to the latency
    and memory it costs, and on the 17-class pilot survey (Protocol~P) we detect no association
    between accuracy and parameter count over a 28-fold range --- a bound set by ten encoders, not a
    demonstration of flatness (Sections~\ref{sec:flat}--\ref{sec:desc}).""",
    "Same null-from-underpowered-test as the Conclusion. Section 4.1's body text already "
    "refuses this inference (12/12 positive cells on Table 2); the contribution list did not.",
)

# ---------------------------------------------------------------------------
# 2. CAVEATS THAT OVERREACH. Table 7's held-out rows are Orange, Coffee and
#    Peach -- exactly configuration A. It carries no data on the crops that
#    define B and C, so it cannot bound the 5.7x figure measured at C.
# ---------------------------------------------------------------------------
edit(
    "contrib1-table7-scope", MAIN,
    r"""Table~\ref{tab:loco} indicates the held-out crops are easier than the trained pool on this
    split, so these multiples are an upper bound rather than a neutral estimate.""",
    r"""Table~\ref{tab:loco} indicates the held-out crops are easier than the trained pool, but its
    held-out rows are Coffee, Orange and Peach --- configuration~A's crops --- so it bounds the
    1.5$\times$ end of this range and says nothing about the crops that define B and C.""",
    "The caveat was applied to the whole 1.5-5.7x range on evidence drawn only from "
    "configuration A. A caveat that outruns its evidence is still an unsupported statement, "
    "and this one understates results the paper may be entitled to.",
)

edit(
    "recipe-attribution", MAIN,
    r"""The 8.5-point spread across that survey is driven by pretraining recipe rather than capacity:
    the two variants of the same 86\,M architecture differ by 4.8 points.""",
    r"""Pretraining recipe accounts for at least 4.8 of the 8.5-point spread across that survey,
    since two variants of the same 86\,M architecture differ by that much; the top of the spread is
    ViT-B-16-SigLIP2, which differs from the MobileCLIP family in architecture and training
    objective as well as recipe, so the remainder is not attributable to recipe alone.""",
    "Evidence covers 4.8 of 8.5 points. The maximum of the spread is a different architecture "
    "with a different objective, not a recipe variant.",
)

edit(
    "fastvit-attribution", MAIN,
    r"""82.2\% as a frozen probe, so the 4.3-point difference is training regime, not architecture.""",
    r"""82.2\% as a frozen probe. The 4.3-point difference is not attributable to architecture, which
    is matched to within 0.7\,M in the same family, but it bundles supervised pretraining, end-to-end
    fine-tuning and a 166-way head against frozen contrastive features with a linear probe, so it
    bounds what regime is worth here rather than isolating one cause.""",
    "'Training regime' bundles three factors, and 10.7M vs 11.4M is family-matched, not identical.",
)

# ---------------------------------------------------------------------------
# 3. THE CATEGORY/COUNT MISMATCH in Section 4.4. Verified against
#    scripts/config.py: LABEL_ALIASES has 5 pairs, EXCLUDE_LABELS has 4 labels,
#    all four of them Wheat (so absent from A). Present per configuration:
#      A  3 pairs, 0 excluded -> 6 alias classes,  6 affected of 16
#      B  4 pairs, 4 excluded -> 8 alias classes, 12 affected of 34
#      C  5 pairs, 4 excluded -> 10 alias classes, 14 affected of 51
#    The published 6/8/10 counts alias classes only, while the sentence names
#    both categories. Density falls monotonically under either reading, so the
#    premise survives; only the accounting was wrong.
# ---------------------------------------------------------------------------
edit(
    "dedup-counts", MAIN,
    r"""Their density falls as the label space grows (6 of 16 evaluated classes at configuration A,
    8 of 34 at B and 10 of 51 at C, or 37.5\%, 23.5\% and 19.6\%), which is the same direction as the
    apparent rise in source-grounded accuracy.""",
    r"""Their density falls as the label space grows. Counting only the classes caught in an alias
    pair, 6 of 16 evaluated classes at configuration A, 8 of 34 at B and 10 of 51 at C (37.5\%,
    23.5\% and 19.6\%); counting the non-disease labels as well, which enter from configuration~B
    because all four are Wheat labels, 6 of 16, 12 of 34 and 14 of 51 (37.5\%, 35.3\% and 27.5\%).
    The direction is the same either way, and it is the same direction as the apparent rise in
    source-grounded accuracy.""",
    "The published counts cover alias pairs only while the sentence names both categories. "
    "Corrected against scripts/config.py. The monotone premise survives both readings.",
)

# ---------------------------------------------------------------------------
# 4. INCOMPLETE ELIMINATION in Section 4.6. Ruling out images-per-class does
#    not rule out total images, which rise 42,326 -> 69,919 (+65%). And the
#    three cells use different class inventories, so they are not commensurable
#    -- the standard Section 3.1 already applies to the zero-shot side.
# ---------------------------------------------------------------------------
edit(
    "seen-trend-elimination", MAIN,
    r"""We report the trend without a mechanism: the obvious explanation, that the added crops bring
    proportionally more training data, is ruled out by Table~\ref{tab:seen}'s own columns, where
    images per class fall from 436 to 421 across the same range.""",
    r"""We report the trend without a mechanism, and one candidate is only partly excluded. Images
    per class do not rise across the range (436, 403 and 421 in Table~\ref{tab:seen}'s own columns),
    so the added crops do not bring proportionally more data per decision; but total training data
    rises 65\%, from 42,326 images to 69,919, and a probe fitted on that much more data would be
    expected to generalise somewhat better on its own. The comparison is also not controlled in the
    sense Section~\ref{sec:data} requires of the unseen side: each configuration is scored on its own
    class inventory, so 78.7\% over 97 classes and 82.2\% over 166 are not measured on the same
    images. The fixed-image-set control named in Section~\ref{sec:limits} would settle this side
    together with the unseen one.""",
    "The elimination addressed images-per-class only; total images (+65%) is the obvious "
    "remaining explanation. Also imports the self-standard Section 3.1 applies to the "
    "zero-shot side but not here.",
)

# ---------------------------------------------------------------------------
# 5. THE NOISE YARDSTICK. Section 4.4 says the 2.4-point SD applies to every
#    Table 2 margin; Section 6 says it does not formally apply to Table 2.
#    Make Section 6 authoritative and state the direction of the bias, which
#    runs in the authors' favour and was not said.
# ---------------------------------------------------------------------------
edit(
    "variance-yardstick", MAIN,
    r"""That interval is the only variance estimate in the article, and it is measured on a
    separately generated registry (Claude Sonnet 5 at temperature 1, against the shipped registry's
    Claude Sonnet 4.6 at temperature 0), so it bounds the order of magnitude of
    descriptor-generation noise without formally applying to Table~\ref{tab:scale}. We use it only in
    that first sense.""",
    r"""That interval is the only variance estimate in the article, and it is measured on a
    separately generated registry (Claude Sonnet 5 at temperature 1, against the shipped registry's
    Claude Sonnet 4.6 at temperature 0), so it bounds the order of magnitude of
    descriptor-generation noise without formally applying to Table~\ref{tab:scale}. We nevertheless
    use it as the yardstick for every margin in that table, our own included, because no better one
    exists; this paragraph, not Section~\ref{sec:dedup}, is the authoritative statement of what it
    licenses. Its likely direction is worth naming because it runs in our favour: sampling at
    temperature 1 should be noisier than the shipped registry's temperature 0, so 2.4 points probably
    overstates the generation noise in Table~\ref{tab:scale} and the margins there may be firmer than
    we treat them.""",
    "Resolves the direct contradiction with Section 4.4 by making one section authoritative, "
    "and states the bias direction, which the paper had not.",
)

edit(
    "variance-determinism", MAIN,
    r"""The per-crop breakdown of Table~\ref{tab:loco} carries bootstrap intervals over images. A
    configuration-C re-measurement on an independently rebuilt embedding cache three weeks later
    reproduced each encoder to within 0.6 percentage points, and the de-duplication re-run of
    Section~\ref{sec:dedup}, executed in a different session on a different machine four months after
    the original sweep, reproduced all twenty uncleaned cells of Table~\ref{tab:scale} exactly, giving
    source-grounded means of 21.51\%, 22.70\% and 23.94\% against the published 21.5\%, 22.7\% and
    23.9\%; both records are released rather than summarised by a single agreement figure.""",
    r"""The per-crop breakdown of Table~\ref{tab:loco} carries bootstrap intervals over images.
    Two further records speak to determinism rather than to variance, and we separate them here so
    they are not read as error bars. A configuration-C re-measurement on an independently rebuilt
    embedding cache three weeks later reproduced each encoder to within 0.6 percentage points, and
    the de-duplication re-run of Section~\ref{sec:dedup}, executed in a different session on a
    different machine four months after the original sweep, reproduced all twenty cells of each
    configuration in Table~\ref{tab:scale} exactly, giving source-grounded means of 21.51\%, 22.70\%
    and 23.94\% against the published 21.5\%, 22.7\% and 23.9\%. Because the descriptor head is
    training-free and the encoders are frozen, exact reproduction is the expected outcome: it shows
    the pipeline is deterministic and the released files regenerate the article, and it places no
    bound at all on how far a margin would move under a fresh registry or a fresh split. Both records
    are released rather than summarised by a single agreement figure.""",
    "Re-execution of a deterministic pipeline is not a variance estimate, and it sat in a "
    "paragraph titled 'Scope and variance' immediately after 'we report no variance "
    "estimates'. Also fixes 'all twenty cells of Table 2' -- Table 2 has 60.",
)

# ---------------------------------------------------------------------------
# 6. SELECTION ON THE OUTCOME OF ONE ARM. The 41-class paired subset is
#    defined by where the grounded arm succeeded, and Section 6 says refusal
#    tracks source availability. That makes +1.96 the optimistic estimate.
# ---------------------------------------------------------------------------
edit(
    "paired-subset-selection", MAIN,
    r"""We therefore evaluate both arms on the intersection of classes that every seed of both arms
    filled genuinely, so that the sourcing requirement is the only difference between them. The
    intersection is smaller than the full held-out set and chance rises accordingly; we report
    both.""",
    r"""We therefore evaluate both arms on the intersection of classes that every seed of both arms
    filled genuinely, so that the sourcing requirement is the only difference between the texts being
    compared. The intersection is smaller than the full held-out set and chance rises accordingly; we
    report both, and we treat the full set as the primary estimate. The reason is that the
    intersection is not selected independently of the arms: every one of the ten excluded classes was
    excluded because the \emph{grounded} arm refused it, and Section~\ref{sec:limits} shows refusal
    tracks how well sourced a disease is. Conditioning on where grounded generation succeeded
    therefore selects classes on which it is most likely to do well, so the paired figure is the
    optimistic reading of the two and the full-set figure the unbiased one.""",
    "Membership in the paired subset is a post-treatment outcome of the arm under test. The "
    "paper explained the +1.96 / +0.94 gap as coverage but never named the selection as bias "
    "in the paired estimate, which the Abstract led with.",
)

edit(
    "abstract-control-order", MAIN,
    r"""A matched ungrounded control over seven seeds isolates sourcing: $+1.96$ points on the paired
    classes and $+0.94$ on the full set, both with intervals including zero, a null rather than
    equivalence.""",
    r"""A matched ungrounded control over seven seeds separates sourcing from authoring: $+0.94$
    points over all 51 held-out classes, and $+1.96$ on the 41 classes grounded generation managed to
    fill --- a subset its own refusals define, so the smaller figure is the unbiased one. Both
    intervals include zero, a null rather than equivalence.""",
    "Abstract led with the selected estimate. Reordered so the unbiased number comes first.",
)

# ---------------------------------------------------------------------------
# 7. 'FIELD-USEFUL' is a claim about the field with no field evidence.
# ---------------------------------------------------------------------------
edit(
    "abstract-field-useful", MAIN,
    r"""Third, a five-item shortlist makes the modest top-1 field-useful: grounded top-5""",
    r"""Third, a five-item shortlist puts the correct disease in front of the operator far more often
    than top-1 does: grounded top-5""",
    "'Field-useful' asserts a property of field practice; no user study, expert assessment or "
    "decision-cost model is reported. The metric statement is what the evidence supports.",
)

edit(
    "contrib3-field-useful", MAIN,
    r"""\item \textbf{Top-5 shortlisting, with a margin-ranked abstain gate}
    makes the modest top-1 field-useful: the shortlist carries the usefulness and the gate
    orders confidence so the system can decline (Section~\ref{sec:abstain}).""",
    r"""\item \textbf{Top-5 shortlisting, with a margin-ranked abstain gate} lifts the chance the
    correct disease reaches the operator from roughly a fifth to roughly two thirds: the shortlist
    carries that gain and the gate orders confidence so the system can decline
    (Section~\ref{sec:abstain}). Whether two thirds is enough to be useful in the field is a question
    about practice that this article does not measure (Section~\ref{sec:limits}).""",
    "Same overclaim, and the honest bound is already computable from Table 3.",
)

edit(
    "abstain-field-useful-body", MAIN,
    r"""The shortlist, not the gate, is what makes the system field-useful; the gate's value is
    that""",
    r"""The shortlist, not the gate, is what carries the accuracy; the gate's value is that""",
    "Third instance of the same unevidenced claim.",
)

# ---------------------------------------------------------------------------
# 8. THE DEPLOYMENT RECOMMENDATION vs the paper's own utility metric. At C the
#    smallest tier's grounded top-5 is 62.9% against 74.6% for MobileCLIP-B,
#    and Section 4.9 shows MobileCLIP-B is the one encoder INT8 speeds up
#    (46.4 ms, 87.4 MB). The paper recommends the smallest tier and reads the
#    shortlist, without ever weighing the two against each other.
# ---------------------------------------------------------------------------
edit(
    "abstract-deployment-rec", MAIN,
    r"""We recommend deploying the smallest tier and reading the shortlist, not the top-1.""",
    r"""We recommend reading the shortlist rather than the top-1, and choosing the tier by objective:
    the 11.4\,M encoder minimises latency and footprint, while the 86.3\,M one, quantised, holds the
    best shortlist accuracy and stays real-time.""",
    "The two halves of the old recommendation pulled against each other: the smallest tier "
    "costs 11.7 points of top-5 at configuration C, and top-5 is the metric the paper says "
    "carries the usefulness.",
)

edit(
    "abstain-deployment-rec", MAIN,
    r"""We therefore recommend it as the deployment setting for large label spaces, which is the
    regime the deployment case actually concerns, and note that at 16 classes the hand-curated bank
    remains competitive.""",
    r"""We therefore recommend it as the deployment setting for large label spaces, which is the
    regime the deployment case actually concerns, and note that at 16 classes the hand-curated bank
    remains competitive. Reading the shortlist also bears on which tier to ship. At configuration~C
    the 11.4\,M encoder reaches 62.9\% top-5 against 74.6\% for the 86.3\,M one, so the smallest tier
    costs 11.7 points of the metric we are recommending be read; and the 86.3\,M encoder is the one
    tier INT8 accelerates (Section~\ref{sec:quant}), to 46.4\,ms at 87.4\,MB, which is still
    real-time. The smallest tier is the right choice under a latency or memory budget and the largest
    under a shortlist-accuracy objective, and we report both rather than collapsing them into one
    recommendation.""",
    "Surfaces the trade-off from the paper's own Tables 3 and 8 instead of asserting a "
    "single sweet spot.",
)

edit(
    "fig7-caption-sweetspot", MAIN,
    r"""marker area is proportional to parameter count and labels give the 8-bit footprint. The
    11.4-million-parameter tier is the deployment sweet spot.}""",
    r"""marker area is proportional to parameter count and labels give the 8-bit footprint. The
    11.4-million-parameter tier minimises latency and footprint; on a top-5 objective the 86.3\,M
    tier is preferable (Section~\ref{sec:abstain}), so the sweet spot depends on which cost
    binds.}""",
    "Caption asserted a single optimum with no stated objective function.",
)

# ---------------------------------------------------------------------------
# 9. INT8: the one encoder that benefits is also the largest, so architecture
#    and scale are not separated, and the proposed confirmation (a 92.9 M
#    transformer) is larger still and would not separate them either.
# ---------------------------------------------------------------------------
edit(
    "int8-scale-confound", MAIN,
    r"""The one pure-transformer encoder in Table~\ref{tab:edge} behaves oppositely, and the
    convolution counts above explain why, but a single row is not enough to state it as a rule:
    adding a second pure-transformer encoder (ViT-B-16-SigLIP2 is already in this study as a
    reference ceiling) is the cheapest way to confirm it.""",
    r"""The one pure-transformer encoder in Table~\ref{tab:edge} behaves oppositely, and the
    convolution counts above explain why, but two limits apply. A single row is not enough to state
    it as a rule; and that row is also the largest model in the table at 86.3\,M against 11.4--35.8\,M
    for the hybrids, so architecture and scale are not separated by this comparison. The convolution
    counts are what make architecture the likelier cause. Confirming it needs a pure transformer
    inside the hybrids' size range, or a hybrid near 86\,M: benchmarking ViT-B-16-SigLIP2, already in
    this study as a reference ceiling, would add a second transformer row but at 92.9\,M would leave
    the confound in place.""",
    "The scale confound was never named, and the remedy the paper proposed does not break it.",
)

# ---------------------------------------------------------------------------
# 10. ROUTING: the Introduction assumes the operator supplies which head
#     applies; Section 6 asserts the image arrives without that information.
#     Both describe the same deployment.
# ---------------------------------------------------------------------------
edit(
    "routing-contradiction", MAIN,
    r"""A field image arrives without a label saying which crop it shows, the probe has no output unit
    outside its 166 classes, and the descriptor head reaches only 9.3\% on those same seen classes,
    so mis-routing is costly in both directions;""",
    r"""Nothing in the pipeline infers which head applies: Section~\ref{sec:intro} takes that signal
    as supplied by the operator, and no component recovers it when it is not. The probe has no output
    unit outside its 166 classes, and the descriptor head reaches only 9.3\% on those same seen
    classes, so mis-routing is costly in both directions;""",
    "Section 1 assumes the routing signal is given; Section 6 asserted it is absent. Recast "
    "around what the pipeline lacks, which is the real gap, so both sentences can hold.",
)

# ---------------------------------------------------------------------------
# 11. Arithmetic. 320.0/62.2 = 5.14x, 614.9/79.5 = 7.73x, 923.8/98.7 = 9.36x.
# ---------------------------------------------------------------------------
edit(
    "qdq-range", MAIN,
    r"""recovers 5--9$\times$ of that but still does not""",
    r"""recovers 5.1--9.4$\times$ of that but still does not""",
    "Computed from Table 8: 5.14x, 7.73x, 9.36x. '5--9' understated the top end.",
)

# ---------------------------------------------------------------------------
# 12. Generated and hand-maintained table captions carrying the same null.
# ---------------------------------------------------------------------------
edit(
    "pilot-caption-null", PILOT,
    r"""Accuracy does not track parameter count (Spearman $\rho = 0.35$, $p = 0.32$).}""",
    r"""No association between accuracy and parameter count is detectable here (Spearman $\rho =
    0.35$, $p = 0.32$); with ten encoders the test resolves only $|\rho| \gtrsim 0.65$, so this bounds
    the association rather than showing it is absent, and Section~\ref{sec:flat} reports a positive
    coefficient in all twelve cells of Table~\ref{tab:scale}.}""",
    "Caption asserted the null from an underpowered test, contradicting Section 4.1's own "
    "refusal to read non-significance as flatness.",
)

edit(
    "supervised-caption-null", GEN,
    r"""_obs = ("Accuracy does not track parameter count: the strongest network is " """.rstrip(),
    r"""_obs = ("Accuracy tracks parameter count weakly at best: the strongest network is " """.rstrip(),
    "Same null-assertion in the generated supervised-baseline caption (rho=0.36, n=14).",
)

edit(
    "loco-caption-scope", GEN,
    '"crops, so difficulty does not follow the held-out boundary. The held-out crops "',
    '"crops, so difficulty does not follow the held-out boundary. The held-out crops here are " '
    '"configuration A\'s three, so this bounds that configuration and not B or C; they "',
    "Table 7's held-out rows are only configuration A's crops, so the contrast cannot bound "
    "the 5.7x figure measured at configuration C. Edited as adjacent Python literals, which "
    "concatenate, so the emitted caption reads as one sentence.",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    texts = {p: p.read_text(encoding="utf-8") for p in {e.path for e in EDITS}}
    statuses: list[tuple[str, str]] = []
    failed = False

    for e in EDITS:
        new_text, status = e.apply(texts[e.path])
        statuses.append((e.ident, status))
        if status in ("applied", "already"):
            texts[e.path] = new_text
        else:
            failed = True

    width = max(len(i) for i, _ in statuses)
    for ident, status in statuses:
        mark = {"applied": "+", "already": "="}.get(status, "!")
        print(f" {mark} {ident.ljust(width)}  {status}")

    if failed:
        print("\nABORTED: at least one anchor is missing or ambiguous; nothing was written.",
              file=sys.stderr)
        return 1

    n_applied = sum(1 for _, s in statuses if s == "applied")
    if args.dry_run:
        print(f"\n--dry-run: {n_applied} edit(s) would be applied to {len(texts)} file(s).")
        return 0

    for p, text in texts.items():
        p.write_text(text, encoding="utf-8", newline="
")
    print(f"\nWrote {len(texts)} file(s); {n_applied} edit(s) applied.")
    if any(e.path == GEN for e in EDITS):
        print("NOTE: make_tex_tables.py changed -- run `python docs/paper/make_tex_tables.py --write`"
              " to regenerate tab_loco.tex and tab_supervised.tex.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
