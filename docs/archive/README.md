# Archive — superseded snapshots

Dated point-in-time context and result dumps, kept for history. **These are NOT the source of
truth** — they contain numbers from earlier, smaller experimental configurations.

| File | Date | Config it describes |
|---|---|---|
| `context_2026-07-01.md` | 1 Jul | early EXP2/EXP3 — 80 seen classes, 3 held crops (17 classes) |
| `results_2026-07-02.md` | 2 Jul | same small config |
| `context_2026-07-13.md` | 13 Jul | 20 crops / 217 classes; probe, bake-off, edge, LOCO |
| `results_2026-07-13_final.md` | 13 Jul | same |

## Why you must not quote these directly

Numbers drifted as the config scaled, so the same metric name means different things:

| Metric | Small config (1–2 Jul) | Full config (14 Jul) |
|---|---|---|
| SEEN accuracy | 67.0% (80 classes) | **82.6%** (166 classes) |
| UNSEEN zero-shot | 27.0% (3 crops, 17 classes, chance 5.9%) | **17.0%** (6 crops, chance ≈2%) |

Both are correct *for their own setup* — the drop is expected as classes increase and chance falls.
The paper must pick **one** canonical configuration and report it consistently.

**Current source of truth:** `docs/paper/results_batch3_run.md` (14 Jul full run) +
`docs/paper/findings_log.md` (evidence trail). Reconciling these into a single results table is
open task **P0-2**.
