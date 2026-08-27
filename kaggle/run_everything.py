"""
DEPRECATED — do not run this. It is the script that burned a 12 h Kaggle session and saved nothing.

Superseded by kaggle/RUN_THIS.py -- one file, the whole study, run it 2-3 times.
See kaggle/RUNBOOK.md. To skip the 114 GB fetch, build the images locally first with
scripts/prepare_upload.py and upload them as the Kaggle dataset  pde-sage-data.

WHAT WENT WRONG HERE
--------------------
This file did everything in one cell with no wall-clock budget, so when the fetch stalled the whole
session was killed by the 12 h cell timeout and committed NOTHING -- including the stages that had
already finished. It spent 11.8 of those hours inside a single request: a shard was asked for at
t+548 s and never returned a byte.

Four causes, all fixed in the replacements:

  1. config.SHARD_REVISION was `refs/convert/parquet`, a floating auto-generated branch that
     HuggingFace REGENERATED when SAGE was reorganised on 2026-08-24 -- three days before the run.
     The job resumed a May-built .shards_done.json against August data. It is now pinned to the
     commit SHA the published results were measured on (config.SAGE_REVISION_MAY).
  2. No HF token, so the pull was rate-limited; a throttled HF connection stalls rather than failing.
  3. hf_hub_download had no enforceable deadline. Shard downloads now run in a child process that is
     killed at 15 minutes and retried.
  4. No budget. Every replacement stops itself at BUDGET_H so the session always commits.

It also assumed 13 shards of a 114 GB release while pulling a 48-shard 21 GB one, and its
MIN_EXPECTED_CROPS of 16 would have quietly accepted a build missing two crops entirely.
"""
import sys

sys.exit(__doc__)
