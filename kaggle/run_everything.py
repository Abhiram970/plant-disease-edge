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

Causes, in order of how well the log supports them:

  1. PROVEN -- hf_hub_download had no enforceable deadline, and its internal backoff will retry a
     dead endpoint indefinitely. Shard downloads now run in a child process killed at 15 min.
  2. PROVEN -- nothing bounded the cell, so the stall took the finished stages down with it. Every
     replacement stops itself at BUDGET_H so the session always commits.
  3. LIKELY, NOT ESTABLISHED -- no HF token. The warning is in the log, but the three shards that
     did arrive came down at 28, 141 and 150 MB/s, so throttling was not limiting throughput.

NOT a cause, though it was first written up as one: a May/August dataset mismatch. This run read the
MAY layout throughout -- its tqdm totals (shard 0000 = 1 batch of 512) match May shard 0's 90 rows,
where August shard 0 holds 14,248 and would have shown 28 batches -- and MAX_SHARDS = 13 was correct
for it. Pinning the revision is still necessary, because `refs/convert/parquet` resolves to August
now and an unpinned run would silently lose Cotton, but that is a forward hazard, not this failure.

Its MIN_EXPECTED_CROPS of 16 would also have quietly accepted a build missing two crops entirely.
"""
import sys

sys.exit(__doc__)
