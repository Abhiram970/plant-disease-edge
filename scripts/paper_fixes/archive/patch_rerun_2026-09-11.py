#!/usr/bin/env python3
"""Revise the scaling claim against the 2026-09-11 de-duplicated re-run.

The run in docs/paper/rerun_2026-09-11/ is the experiment Section 6 previously named as
missing: `evaluate.py --clean` at configurations A, B and C on all five encoders, plus
the as-published sweep re-measured with per-class metrics. It settles three questions
that the manuscript had bundled into one claim, and they have different answers.

  Q1  Does grounded rise monotonically with the label space?
      Only under the uncleaned label set AND micro-averaging -- one of the four
      combinations of {uncleaned, de-duplicated} x {micro, class-macro}. FALSIFIED.

  Q2  Does the keyword bank decay with the label space?
      Falls monotonically in all four. SURVIVES.

  Q3  Does grounded overtake the bank at the largest scale?
      Leads at C in all four, and the margin widens as the analysis gets more
      conservative: +2.42 as published, +4.93 de-duplicated under class-macro. SURVIVES.

A fourth result falls out: under class-macro the bank's advantage at the two smaller
scales nearly vanishes (-0.16 and -1.22 uncleaned; -0.71 and -0.41 de-duplicated), so
its early lead is concentrated in a few large classes that micro-averaging rewards.

Headline numbers stay uncleaned and micro: cleaning RAISES every figure, so reporting
the uncleaned set remains the conservative choice, and the de-duplicated run is reported
as the control that decides the trend rather than as a new headline.

    python scripts/paper_fixes/patch_rerun_2026-09-11.py --dry-run
    python scripts/paper_fixes/patch_rerun_2026-09-11.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "docs" / "paper" / "tex" / "main.tex"


def _squash(s: str) -> str:
    return " ".join(s.split())


class Edit:
    def __init__(self, ident: str, old: str, new: str, why: str) -> None:
        self.ident, self.old, self.new, self.why = ident, old.strip("\n"), new.strip("\n"), why

    def status(self, text: str) -> str:
        if _squash(self.new) in _squash(text):
            return "already"
        n = text.count(self.old)
        if n == 1:
            return "apply"
        if n > 1:
            raise SystemExit(f"[{self.ident}] anchor ambiguous ({n} matches)")
        raise SystemExit(
            f"[{self.ident}] anchor not found and replacement absent -- source drifted.\n"
            f"--- expected ---\n{self.old[:500]}")


EDITS: list[Edit] = []


def edit(ident: str, old: str, new: str, why: str) -> None:
    EDITS.append(Edit(ident, old, new, why))


# ---------------------------------------------------------------------------
edit(
    "R1-abstract",
    r"""source-grounded text reaches 23.9\% having risen monotonically. Part of that crossover may
be label cleanliness: de-duplicating the largest configuration adds 6.9 points.""",
    # Wording as it stands after the abstract was trimmed back under CEA's 250-word limit;
    # the earlier, longer draft of this sentence pushed it to 299.
    r"""source-grounded text reaches 23.9\%. A de-duplicated re-run at all three scales shows the bank's
decay survives cleaning and every averaging convention while the apparent rise of grounded text
does not, though grounded still leads at 51 classes in all four, by 2.4--4.9 points.""",
    "The monotone rise is falsified 3 of 4 ways; the crossover survives 4 of 4 and widens.",
)

edit(
    "R1-contribution-2",
    r"""single generic symptom sentence does not --- it loses to the bare name at 34 and 51 classes --- and
source-grounded descriptors keep improving as the unseen label space triples, a trend
Section~\ref{sec:limits} reports as provisional because it cannot yet be separated from falling
duplicate-label density. We further show that""",
    r"""single generic symptom sentence does not --- it loses to the bare name at 34 and 51 classes.
Source-grounded text is also the only strategy that does not degrade as the unseen label space
triples: a de-duplicated re-run at all three scales (Section~\ref{sec:limits}) shows its apparent
monotone rise is an artefact of the uncleaned label set, while the keyword bank's decay is not,
and source-grounded text holds the largest scale under every averaging convention and label set we
measured. We further show that""",
    "Contribution 2 claimed the rise; the defensible claim is non-degradation.",
)

edit(
    "R1-averaging-convention",
    r"""92.9\,M reference encoder. The class distribution is capped but not balanced, so we also
checked the scaling result under crop-macro averaging, which gives 21.9\%, 23.8\% and
23.9\% at configurations A, B and C against the micro values of 21.5\%, 22.7\% and
23.9\%; the monotone rise holds under either convention.""",
    r"""92.9\,M reference encoder. The class distribution is capped but not balanced --- classes run
from the 25-image floor to the 600-image cap --- so we report two macro conventions alongside it.
Crop-macro averaging gives 21.9\%, 23.7\% and 23.9\% for source-grounded text at configurations A,
B and C against micro values of 21.5\%, 22.7\% and 23.9\%. Class-macro averaging, the convention a
reader concerned with rare diseases should prefer, gives 27.5\%, 23.3\% and 25.5\%. The three
conventions agree on which strategy wins at each scale and disagree on the shape of the
source-grounded trend, which is why Section~\ref{sec:limits} settles that question with a
de-duplicated re-run rather than with a single curve.""",
    "Class-macro was never reported, and it is the convention that breaks the monotone claim.",
)

edit(
    "R1-section-4.3-scaling",
    r"""\paragraph{Source-grounded descriptors keep improving with scale} As the unseen label space grows
from 16 to 51 classes, source-grounded descriptors improve monotonically (21.5\%, 22.7\%, 23.9\%),
and at configuration C they are the strategy that most clearly beats a bare class name. The generated
registry covers 156 of 217 classes under a uniform schema.""",
    r"""\paragraph{Source-grounded descriptors do not degrade with scale; the keyword bank does} As the
unseen label space grows from 16 to 51 classes, source-grounded text moves 21.5\%, 22.7\%, 23.9\%
under the headline convention while the keyword bank falls 28.9\%, 25.0\%, 21.5\%. We resist
reading the first sequence as a rise. Section~\ref{sec:limits} reports a de-duplicated re-run at
all three scales which shows that ordering is not stable: source-grounded text rises monotonically
only under the uncleaned label set with image-weighted averaging, and under de-duplicated labels,
under class-macro averaging, or under both, it is flat to slightly falling. What is stable across
all four combinations is the comparison --- the bank falls monotonically in every one of them,
source-grounded text falls far more slowly or not at all, and it is the best of the four strategies
at 51 classes in every one. The claim we make is therefore about robustness to a widening label
space rather than about improvement. The generated registry covers 156 of 217 classes under a
uniform schema.""",
    "The 'keeps improving' paragraph is the claim the re-run falsifies.",
)

edit(
    "R1-fig1-caption",
    r"""\caption{Mean zero-shot accuracy over the four deployable encoders as the unseen label space grows.
Source-grounded descriptors improve as the label space widens. The keyword-retrieved bank is shown
for reference only: its prototypes collide (Section~\ref{sec:desc}), so the gap between the two is
not a measurement of grounding.}""",
    r"""\caption{Mean zero-shot accuracy over the four deployable encoders as the unseen label space
grows. The keyword-retrieved bank decays as the label space widens while source-grounded text does
not; the de-duplicated re-run of Section~\ref{sec:limits} shows the bank's decay is robust to
label cleaning and averaging convention whereas the apparent rise of source-grounded text is not.
The bank is shown for reference only: its prototypes collide (Section~\ref{sec:desc}), so the gap
between the two is not a measurement of grounding.}""",
    "Caption asserted the rise the re-run falsifies.",
)

edit(
    "R1-limitations-resolved",
    r"""de-duplicated evaluation at configuration C, released as \texttt{zeroshot\_eval\_C\_clean.json}:
removing one label from each of the five alias pairs, and four of the five non-disease
labels, leaves 42 classes and raises the grounded mean from 23.9\% to 30.8\%. Part of that
6.9-point rise is mechanical: dropping nine classes lifts the chance floor from 1.96\% to 2.38\%,
so relative to chance the cleaned figure is 12.9$\times$ against 12.2$\times$ for the uncleaned
one, and the gain net of the smaller label space is far below 6.9 points. Cleaning still helps the
method on either reading, so reporting the uncleaned numbers as the headline is conservative, but the equivalent runs at A and B do not yet exist and the scaling claim
should be read as provisional until they do. We recommend excluding both the rating labels and the
alias pairs in future use of this benchmark.""",
    r"""de-duplicated evaluation at all three configurations, released as
\texttt{zeroshot\_eval\_\{A,B,C\}\_clean.json}. Removing one label from each of the five alias
pairs and four of the five non-disease labels leaves 13, 26 and 42 classes, and raises the
source-grounded mean to 29.3\%, 32.8\% and 30.8\% against the uncleaned 21.5\%, 22.7\% and 23.9\%.
Part of each rise is mechanical, since dropping classes lifts the chance floor --- at configuration
C from 1.96\% to 2.38\%, so relative to chance the cleaned figure is 12.9$\times$ against
12.2$\times$. Cleaning helps every strategy, so reporting the uncleaned numbers as the headline
remains the conservative choice.

The re-run answers the question this subsection previously left open, and it answers it against
us on one count of three. Source-grounded accuracy rises monotonically only under the uncleaned
label set with image-weighted averaging; de-duplicated it goes 29.3\%, 32.8\%, 30.8\%, and under
class-macro averaging it falls in both label sets. We therefore withdraw any claim that
source-grounded descriptors improve as the label space grows. What the re-run leaves standing is
the comparison: the keyword bank falls monotonically in all four combinations of label set and
averaging convention, by between 6.0 and 12.5 points, source-grounded text does not, and it is the
strongest of the four strategies at the largest scale in all four --- leading the bank by 2.4
points as published and by 4.9 points de-duplicated and class-averaged, so the margin widens as the
analysis becomes more conservative rather than shrinking. One further asymmetry is worth stating
because it runs in our favour and we did not look for it: under class-macro averaging the bank's
advantage at the two smaller scales nearly disappears (0.2 and 1.2 points uncleaned, 0.7 and 0.4
de-duplicated), so the lead it holds under image-weighted averaging is concentrated in a few large
classes. Every one of these margins is within the 2.4-point seed-to-seed standard deviation
measured by the control arm, so we report the direction and its consistency across four
combinations, not the size. We recommend excluding both the rating labels and the alias pairs in
future use of this benchmark.""",
    "The experiment Section 6 named as missing now exists; report what it found.",
)

edit(
    "R1-scope-reproduction",
    r"""configuration-C re-measurement on an independently rebuilt embedding cache three weeks later
reproduced each encoder to within 0.6 percentage points; both records are released rather""",
    r"""configuration-C re-measurement on an independently rebuilt embedding cache three weeks later
reproduced each encoder to within 0.6 percentage points, and the de-duplication re-run of
Section~\ref{sec:limits}, executed in a different session on a different machine four months after
the original sweep, reproduced all twenty uncleaned cells of Table~\ref{tab:scale} exactly, giving
source-grounded means of 21.51\%, 22.70\% and 23.94\% against the published 21.5\%, 22.7\% and
23.9\%; both records are released rather""",
    "A cross-session exact reproduction is evidence a reviewer will value; it was free.",
)

edit(
    "R1-conclusion",
    r"""every other experiment. The levers are, in order: the descriptor authoring method --- source-grounded text is worth
8.8 points over a bare class name at the focused scale, and is the only strategy that does not
decay as the label space triples, a trend we report as provisional because
Section~\ref{sec:limits} cannot yet separate it from falling duplicate-label density --- honest
abstention, and the frozen-backbone hybrid.""",
    r"""every other experiment. The levers are, in order: the descriptor authoring method ---
source-grounded text is worth 8.8 points over a bare class name at the focused scale, and a
de-duplicated re-run at all three scales shows it is the only strategy that does not decay as the
label space triples, though not one that improves --- honest abstention, and the frozen-backbone
hybrid.""",
    "Conclusion still carried the provisional rise.",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    text = MAIN.read_text(encoding="utf-8")
    plan = [(e, e.status(text)) for e in EDITS]
    for e, st in plan:
        if st == "apply":
            text = text.replace(e.old, e.new, 1)

    if not args.dry_run:
        with open(MAIN, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)

    n = sum(1 for _, s in plan if s == "apply")
    print(f"{'would apply' if args.dry_run else 'applied'} {n}/{len(EDITS)} edits\n")
    for e, st in plan:
        print(f"  [{st:7s}] {e.ident:26s} {e.why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
