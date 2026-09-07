# Revision roadmap — Compact VLMs for cross-crop plant-disease diagnosis at the edge

Ordered by what a reviewer would stop on first. Section ordering and document
structure were excluded from this audit at the author's request.

Counts: **11 major**, **13 moderate**, **5 minor**. Eight items are gate blockers.

---

## Tier 0 — must be fixed before the file leaves your machine

These are mechanical, unambiguous, and cost hours rather than experiments.

1. **Repoint Figure 8 at `wiseft.json`.** `make_figures.py:79` still reads
   `run_all_exp3_lw11_full.json`, the 17-unseen-class pilot you already retired
   (`wiseft.json`'s own `note` field records the retirement). Table 4 and
   Figure 8 currently disagree on every value, on the number of sweep points,
   on the unseen class count printed in the figure subtitle, and on which
   alpha to recommend. Then rewrite Figure 8's caption: "Full fine-tuning
   nearly halves unseen accuracy" is a statement about the old data and
   contradicts your own Section 4.5.

2. **Fix the table overflows.** Pages 4 and 6 are unreadable as typeset.
   Move Table 1, Table 2 and Table 5 to `table*` (full-width float), or drop
   columns. Table 2's caption defines `acc@cov X` and those columns are
   clipped off the page; Table 7's `INT8 (MB)` column does not render at all,
   which is the column Contribution 4 promises ("real INT8 sizes") and the one
   the abstract's 12.9 MB comes from. Recompile and read every table page
   before resubmitting.

3. **Mint the Zenodo DOI.** `PENDING-ZENODO-DOI` is still in the
   data-availability statement. Option C rules will not clear without it.

4. **Reconcile Table 3 and Table 4.** Table 3 says the frozen probe is 82.2%
   on 166 seen classes (69,919 images); Table 4's alpha=0 "frozen" row says
   59.0% on 166 seen classes (23,445 images), and its caption asserts that
   alpha=0 "reproduces the frozen baseline". Either explain the different
   evaluation set in the caption or rerun the sweep against the Table 3 probe.

5. **Trim the abstract to 250 words** (currently 325) and fix the two
   misstated ranges while you are in there: "averaged over the four deployable
   tiers, 2.2-13.9x" is a min-max over individual encoder x configuration
   cells (averaged over tiers it is 3.4x-12.2x), and "the 11.4 M tier alone
   spans 2.9-11.3x" matches no descriptor strategy in your result files
   (grounded gives 2.82x-11.04x).

---

## Tier 1 — the claims that decide the paper

6. **Report the full-51 sourcing comparison.** Your headline is +1.96 points,
   "positive on all four deployable encoders". Recomputed from
   `zeroshot_eval_C_gmseeds.json` and `zeroshot_eval_C_ungseeds.json` — same
   seven seeds, same long descriptions — the full 51-class set gives +1.89,
   +0.36, +1.69 and **-0.19**: mean +0.94, one encoder negative. Your own
   length-matched control also has MobileCLIP-B negative. The 41-class pairing
   is methodologically defensible and you argue for it well; the problem is
   that the full-set number is never shown. Report both, and soften the
   abstract, Section 5 and Highlight 2 to match. Suggested wording: *"worth
   +1.96 points on the 41 classes both arms filled, and +0.94 points on the
   full held-out set, where the largest encoder reverses."*

7. **Withdraw the length attribution in Section 4.3.** The paper reads the
   +1.96 -> +1.13 shrinkage as evidence that "the sourcing advantage is not
   independent of how much of each description the encoder actually reads."
   The arithmetic says otherwise: at constant length, 41 -> 51 classes moves
   it +1.96 -> +0.94; at constant class set, long -> 50-word cap moves it
   +0.94 -> **+1.13**, i.e. shortening slightly *raised* it. The comparison as
   published changes two variables at once.

8. **Disclose the duplicate labels, and bound their effect.** Configuration A
   contains three synonym pairs — Orange|Canker / Orange|Citrus_Canker,
   Orange|Greening_Disease / Orange|Huanglongbing, Peach|Leaf_Curl /
   Peach|Peach_Leaf_Curl — so 6 of 16 classes (37.5%) sit in a pair no
   descriptor can separate. Configuration C adds two more pairs, giving about
   19.6%. Duplicate density therefore *falls* monotonically A -> B -> C, which
   is the same direction as your headline "source-grounded descriptors keep
   improving as the label space triples". A reviewer who notices this has a
   mundane alternative explanation for your central result. You need either a
   duplicate-free rerun of all three configurations or an explicit bound on
   how much of the 21.5 -> 23.9 rise the cleanup accounts for.

9. **Surface `zeroshot_eval_C_clean.json`.** You have already run the cleaned
   configuration C — 42 classes, 5 alias pairs merged, 4 labels excluded — and
   it gives grounded means of 30.8% against the 23.9% you report. The
   Limitations section recommends the exclusion as if it had not been done.
   Publishing it costs you nothing (the numbers are better) and removes the
   appearance of having chosen the label space that suits the trend. Run the
   A and B equivalents so the scaling claim can be checked clean.

10. **Fix Table 6's caption and Contribution 5.** The held-out crops rank
    1st, 2nd and 4th of six and average 19.0% against 9.0% for the trained
    crops. "Held-out crops fall in the middle of the range" is not what the
    table shows, and Section 4.6 already concedes the opposite two paragraphs
    later. Also state that this analysis uses `rich` descriptors on a
    78-class protocol — a different strategy and a different label space from
    the headline result.

11. **Qualify or rerun the supervised sweep.** All fourteen runs stop at four
    epochs; twelve are still improving at the last epoch (up to +3.4 points
    between epochs 3 and 4). "All architectures converge" in Figure 9's
    caption is not true, and a fixed short budget systematically favours the
    fast-converging small models whose success is the basis for "capacity is
    not the lever" and Spearman rho = 0.36. Two models (fastvit_t8,
    mobilenetv3_small_100) are reported at an epoch-4 value *below* their own
    epoch-3 peak, so the convention is not even uniform. Either train to
    convergence or state the budget as a limitation and stop drawing a
    capacity conclusion from it.

12. **Fix the "any full symptom description beats a class name" claim.**
    Averaged over the four deployable encoders, `crude` scores 14.1 / 18.7 /
    16.7% against `bare`'s 12.7 / 20.3 / 19.1% — crude *loses* at B and C.
    The crude column is also the only one in Table 1 without a mean row
    (`make_tex_tables.py:94` hardcodes the dash), so the statistic that would
    expose this is the one omitted. Restore the mean row and narrow the claim
    to source-grounded text.

---

## Tier 2 — credibility repairs

13. **Separate the 82.2% / 9.3% pair.** 82.2% is the 166-class probe; 9.3% is
    the 80-class pilot run. `make_figures.py:71-73` says so in a comment. The
    matched pair inside that file is 67.2% against 9.3%.

14. **Label Figure 1 and Section 4.2 as pilot-protocol.** Figure 1's y-axis
    already says "(17 classes)"; the caption and the Section 4.1 sentence that
    cites it do not. Section 4.2's "beats the larger 21.5 M model" reverses on
    the main protocol — MobileCLIP-S1 beats MobileCLIP2-S0 at all three
    configurations with grounded descriptors — and your own `findings_log.md`
    calls that pilot ranking "noisy". The "86% of the reference" figure is
    76% at configuration C.

15. **Resolve the SCOLD contradiction.** `findings_log.md` marks the 5.3%
    result "not valid (RoBERTa from base; preprocess mismatch)". Section 4.2
    disclaims it properly, but Related Work and Figure 3's caption both state
    it flatly, two pages earlier. Make the disclaimer travel with the claim.

16. **Reconcile Section 6 with Section 4.3.** Section 6 says no variance
    estimates are reported apart from the LOCO bootstrap; Section 4.3 reports
    seed standard errors and a 95% confidence interval.

17. **Support or drop the 0.15-point reproducibility claim.** The only other
    configuration-C record in the repo (`run_2026-09-01_reference.md`) differs
    from the current JSONs by 0.1-0.6 points per encoder, and that file itself
    says a disagreement is worth investigating. Name the two runs and release
    both.

18. **Measure INT8 accuracy or stop recommending INT8.** `findings_log.md`:
    "Calibration used random tensors — valid for latency; an INT8 *accuracy*
    claim needs `--calib-dir` with real crop images." Section 4.8 issues a
    deployment rule without an accuracy number anywhere in the paper.

19. **Make `SOURCE_CHECKLIST.md` reconcile with the Limitations counts.** All
    175 URLs are marked `[not checked]`; the released artifact is URL-indexed
    while the paper's counts (156 / 112 / 40 / 16 / 47 / 23) are
    record-indexed. Auditability is your differentiator, so this is the one
    artifact that has to add up.

20. **Retune "flat with size".** Two models of near-identical size differ by
    8.5 points in Figure 1, and Table 1's configuration-C grounded column
    spans 21.6-28.4%. You call that flat while treating +1.96 as real and 4.3
    points as "training regime, not architecture". The supportable claim is
    that *parameter count does not predict accuracy over this range*.

21. **Fix Figure 5's caption.** At configuration B the hand-curated bank beats
    source-grounded text on three of four deployable encoders, so "leads only
    at the smallest" is wrong at the middle scale.

22. **Correct "modest (17-29%)".** Grounded top-1 spans 13.7-28.6% across your
    deployable tiers; 13.7% is printed in both Table 1 and Table 2. The
    paragraph is headed "Honesty on absolutes".

---

## Tier 3 — polish

23. Page footers on the figure pages read "Page 9 of 8" and "Page 10 of 8".
24. Table 6's caption exposes the internal checkpoint tag `dfndr2b`.
25. Twenty references, none from the target journal, and a one-column Related
    Work section. Thin for a special issue; an editor screening for fit will
    notice.
26. Five paragraphs run 188-229 words (tex lines 124, 184, 258, 600, 682).
27. 70 em-dashes in ~5,500 words. Stylistic only, but the density is high.

---

## What is already solid — do not touch

- The bibliography is clean: 20 entries, all cited, none missing, none
  orphaned. `check_references.py` reports eight "undefined reference"
  criticals; **all eight are false positives** — the labels live in the
  `\input`-ed `tab_*.tex` files and the checker does not follow `\input`.
- Table 1, Table 2, Table 3, Table 5, Table 6 and Table 7 reproduce their
  source JSONs exactly. Every value checked matched.
- Section 4.6's arithmetic is correct throughout: 7.1, 2.5, 9.2, 4.3 points,
  10x, 25x, and Spearman rho = 0.358 -> 0.36 all verify.
- The descriptor-collision analysis in Section 3.4 (8 unique / 17 no-match /
  26 collided at C; coverage 68.8 / 61.8 / 66.7%; largest groups of seven and
  five) matches `descriptor_coverage.json` exactly.
- The quantisation analysis verifies: 18-19x slower dynamic, 5-9x static
  recovery, 2.2x faster on the pure transformer, 119/215/235 unconverted
  convolutions against 3.
- The paired-41 control-arm statistics (+1.76/+2.00/+2.15/+1.93, mean +1.96,
  CI [+1.70, +2.22]) and the length-matched control
  (+1.26/+1.96/+1.69/-0.38, mean +1.13) are both computed correctly.
- Highlights are within Elsevier's 85-character limit.
- The Limitations section is unusually candid, and the decision to report the
  dirty 51-class numbers rather than the flattering cleaned ones is
  conservative in the right direction. Most of the fixes above are about
  making the rest of the paper as honest as Section 6 already is.
