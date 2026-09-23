# Audit history and recurrence ledger

Written 2026-09-12 after reading every prior audit. Purpose: stop each new audit from
rediscovering the same things, and record *why* new issues keep appearing.

**Read this before commissioning another audit.**

## Where the audits are

Five passes were archived **outside the repo** by `scripts/paper_fixes/archive_stale.py`,
which is why an in-repo search finds nothing:

| # | Date | Pass | Location | Outcome |
|---|---|---|---|---|
| 1 | 2026-09-08 | deep review | `../plant-disease-edge-archive/audits/AUDIT_2026-09-08.md` | 3 blockers, 6 major, 9 moderate |
| 2 | 2026-09-08 | script workspace | `../plant-disease-edge-archive/audits/logs/audits/2026-09-08/main-2/` | Desk Reject, 4.0/10 |
| 3 | 2026-09-09 | re-audit | `../plant-disease-edge-archive/audits/AUDIT_2026-09-09.md` | 18 resolved, 6 open, 3 new |
| 4 | 2026-09-09 | fresh pass | `../plant-disease-edge-archive/audits/AUDIT_2026-09-09_fresh_pdf.md` + `_fresh_issues.json` | 49 issues (12 major) |
| 5 | 2026-09-09 | third pass | `../plant-disease-edge-archive/audits/AUDIT_2026-09-09_pass3.md` | 1 open (DOI) |
| 6 | 2026-09-11 | deep review | `docs/paper/review_results/main/` | 21 issues (8 major) |
| 7 | 2026-09-12 | deep review | `docs/paper/review_2026-09-12/main/` | 26 issues (7 major) |
| 8 | 2026-09-12 | re-audit | `docs/paper/review_2026-09-12-post/main/` | all prior resolved |
| 9 | 2026-09-22 | deep review | `docs/paper/review_2026-09-22/main/` (not committed) | 29 issues (6 major) |

Fix rounds: `patch_audit_2026-09-10.py`, `patch_rerun_2026-09-11.py`,
`patch_audit_2026-09-12.py` and, for pass 9, `apply_revision_text.py` +
`apply_revision_results.py`, all in `scripts/paper_fixes/`.

## The items that recur, and why

### 1. Placeholder DOI — flagged in **6 of 8 passes**, never fixed
`B1` in passes 1, 3 and 5 (where it was *the only* open item), `placeholder-doi` in 6,
`pending-zenodo-doi` in 7. `preflight.py` fails on it every run.

**Why it recurs:** only the authors can mint a Zenodo DOI. No amount of editing closes it.
**Action:** mint it. Until then every audit will open with it and it will read as the
paper's headline defect, because for a reproducibility-forward paper it is one.

### 2. Caption / figure claims disagreeing with the tables — **5 passes, ~10 instances**
Pass 1 B3 (Figure 4 contradicted Table 2 on every value and asserted a retracted claim);
pass 3 NEW-3 (Pareto title contradicted its own plotted data); pass 4 (Figure 10 matched
no table and inverted the encoder ranking; Figure 6 legend contradicted Table 2; Figures 1
and 3 disagreed on the same encoders; Figure 1 labelled two distinct points identically);
pass 6 (`fig3-caption-contradicts-plot`, `withdrawn-claim-left-in-captions`); pass 7
(`fig7-caption-sweetspot`, `pilot-caption-null`, `loco-caption-scope`).

**Why it recurs — this is the structural one.** Claims live in three places that are never
cross-checked against the data: prose in `tex/main.tex`, a hand-maintained caption in
`tex/tab_pilot.tex`, and generated captions inside `make_tex_tables.py`. `preflight.py` has
14 checks and **not one of them reads a caption**. Numbers are protected by
`make_tables.py`; the sentences *about* the numbers are not.

**Action:** the highest-value gate left to build. A check that every caption claiming a
direction ("does not track", "rises", "sweet spot", "flat") names the artifact it is read
from would have caught 10 findings across 5 audits.

### 3. Non-significance asserted as a positive finding — **4 passes**
Pass 1 D4, pass 4 ("a non-significant result is used to support an equivalence claim"),
pass 6 (`size-flat-claim-vs-own-tables`), pass 7 (`null-from-underpowered-test`).

**Why it recurred:** each round fixed the *body text* and left the claim standing in the
caption, the contribution list and the Conclusion. Section 4.1 has had the correct careful
version since pass 3 while three other locations contradicted it.
**Fixed 2026-09-12** in all four locations at once. Watch for reintroduction.

### 4. Single-seed headline tables / variance — **4 passes**
Pass 1 C1 (wrong replication unit: encoders, n=4, CI [+1.70,+2.22] → seeds, n=7,
[−0.23,+4.14]), pass 4, pass 6 (`single-seed-headline-vs-measured-noise`), pass 7
(`determinism-not-variance`).

**Why it recurs:** Tables 2–6 are genuinely single-seed. Disclosure is not resolution.
**Action:** regenerate the registry over ≥3 seeds at A/B/C, or accept it as a permanent
limitation and stop expecting audits to stop mentioning it.

### 5. Missing DCLIP / CuPL baseline — passes 6 and 7
Cannot be closed by editing. Text-tower only; no new images or training.

### 6. Em dashes — raised by the script layer in **every** pass
Pass 3 explicitly ruled on it: *"E7 — em dashes: still advisory, still ignore."* The script
layer has re-raised it every run since, 10× per run in pass 7.
**Action:** it is noise for an Elsevier submission. Count is back to 10 as of 2026-09-12.
Ignore permanently unless a style guide forbids them.

## Why new issues appear every time

Four mechanisms, in order of how much they explain:

1. **Fixes were made by adding hedges, and every hedge is a new claim with a new surface.**
   This is the dominant mechanism and it is self-sustaining. Traceable chains:
   - Pass 1 C1 (wrong replication unit) → fix introduced the "2.4-point standard deviation"
     language → pass 7 found §4.4 and §6 asserting *contradictory* things about what that
     number licenses.
   - Pass 6 `table4-mechanism-contradicted` → fix added the images-per-class elimination →
     pass 7 found the elimination incomplete (total images rise 65%, unaddressed).
   - Pass 6 `abstract-missing-conclusion-element` → fix added a deployment recommendation →
     pass 7 found the recommendation costs 11.7 points of the paper's own utility metric.
   - Pass 6 `no-seen-unseen-routing-evaluation` → fix added Section 6 wording → pass 7
     found it contradicting the Introduction's operating assumption.

   **Rule going forward: prefer deleting an overreaching claim to qualifying it.** A caveat
   is a claim and will be audited as one. The 2026-09-12 round was written this way.

2. **~30% of every bundle is three items prose cannot fix** (DOI, missing baselines,
   single-seed). They will be re-found every pass. Track them as known-open, not as new.

3. **The script layer emits ~300 low-signal items per run** (em dashes, long-paragraph
   heuristics, 289 "visual" findings in pass 7) which swamp the substantive findings and
   make each bundle look bigger and newer than it is.

4. **Internal sources went stale and fed contradictions back in.** `findings_log.md` still
   asserted "source-grounded descriptors keep improving" — withdrawn in §4.4 — plus stale
   numbers for runs 13, 14 and 15, four months after the manuscript moved on. Now carries a
   supersession block at the top. **Any doc that states a claim needs the same treatment.**

## Genuinely settled — do not re-litigate

From pass 3's integrity checks and passes 6–7: the frozen-backbone decision and its
negatives; the collision defect behind the retracted `rich` column; the INT8 /
float-convolution mechanism; protocol fencing between N and P; the nested-design caveat;
the LOCO rename; abstract baselines; the withdrawn scaling claim. Pass 8 re-verified all
24 fixes from pass 7 in the rebuilt PDF.

## Pass 9 (2026-09-22) and what it changed

Pass 9 audited the salvaged manuscript, so most of its findings are new rather than recurrences:
sentences written for the old "grounded wins" framing had survived the reversal of the result.

Closed by **editing** (`apply_revision_text.py`, 43 guarded replacements into `docs/paper/manuscript/revision/`):
the under-disclosed DCLIP+/CuPL+ prompts, the inverted "conservative direction", register asserted
as a demonstrated cause, the unconditional authoring-beats-size claim, the edge claim on an x86
laptop, and eleven smaller items.

Closed by **measuring** (`scripts/rescore_descriptors.py`, then `apply_revision_results.py`):

| open item | what the re-score found |
|---|---|
| register never tested | stripping taxonomy moves accuracy +4.4 / -2.0 / -2.1 at A/B/C, and +0.2 / -2.7 / -3.7 class-balanced: the register reading is **not** supported |
| image-informed prompt may inflate the margin | on foliar classes only the CuPL+ margin at C is 6.3 against 7.0, so the instruction buys at most ~1 point |
| top-5 and abstention unmeasured for the recommended registry | CuPL+ top-5 72.7-81.4% at C against 59.0-74.6% for grounded |
| micro-averaging hides class imbalance | class-macro reported at every cell; CuPL+ leads at every configuration on both label sets |

**Still open and only the authors can close it:** the Zenodo DOI, now flagged by 7 of 9 passes.

The pass-9 workspace is not committed (see `.gitignore`); the revision it produced is.
