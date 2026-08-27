# Kaggle runbook — two runs, both well under the 12 h limit

The previous single-cell attempt burned 12 hours and committed **nothing**. This splits the work so
no run can do that again, and so a run that *is* cut short still saves everything it finished.

| | what | accelerator | wall | needs |
|---|---|---|---|---|
| **local** | `scripts/prepare_upload.py` → upload `pde-sage-data` | — | ~25 min + upload | nothing |
| **run 2** | zero-shot, abstention, clean eval, **ungrounded arm**, probe, LOCO | GPU T4×2 | ~5–7 h | `pde-sage-data` |
| **run 3** | 14 supervised CNN baselines | GPU T4×2 | ~7–8 h | `pde-sage-data`, `pde-results-2` |

`run1_data.py` exists but you should not need it — see [Run 1](#run-1-fallback-only) below.

---

## Why the last run died

It spent **11.8 of its 12 hours inside a single call**: a shard was requested at t+548 s and had
still not returned one byte when the notebook was killed. Four independent causes, all now fixed:

1. **A floating dataset revision.** `SHARD_REVISION` was `refs/convert/parquet` — an auto-generated
   branch that HuggingFace **regenerated** when SAGE was reorganised on **2026-08-24**, three days
   before the run. The job was resuming a May-built `.shards_done.json` against August data.
   Now pinned to a commit SHA.
2. **No token.** The log carried *"You are sending unauthenticated requests to the HF Hub"*.
   Anonymous pulls are rate-limited, and a throttled HF connection does not fail — it stalls.
3. **No deadline.** `hf_hub_download` has no usable wall-clock bound and its internal backoff will
   retry a dead endpoint indefinitely. Shard downloads now run in a child process that is killed at
   15 minutes and retried, so a bad shard costs minutes rather than a session.
4. **No budget.** A cell that overruns is killed and commits nothing. Every run now stops itself at
   `BUDGET_H = 8.5` and exits cleanly with whatever it has.

### The part that matters for the paper, not just the run

SAGE shipped two incompatible releases:

| release | commit | shards | size | held-out crops | classes at scale C |
|---|---|---|---|---|---|
| **2026-05-07** | `bc9bd2899f` | 13 | 114 GB | **8** | **51** |
| 2026-08-24 | `dde0de8633` | 48 | 21 GB | 7 | 48 |

The August release is **not a superset**. Its own `canonical_mapping.json` marks all 14 Cotton
entries `"how": "no-canonical-crop"`, and the crop column of all 48 August shards contains **zero**
Cotton rows. Every published number here was measured with Cotton in the held-out set, so
`config.py` is pinned to **May**, and `prepare_upload.py` reproduces exactly the published split:

```
scale A   3 held crops   16 classes
scale B   6 held crops   34 classes   (incl. Cotton ×3)
scale C   8 held crops   51 classes   (incl. Cotton ×3)
seen                    166 classes
```

Do not repoint the pin without re-measuring every zero-shot number. `PDE_SAGE_REVISION` overrides it
if you ever want the August release as a robustness check.

---

## Local step — build and upload the dataset

The images are already on this machine at full resolution (60.8 GB, up to 3072 px). Downscaling them
locally and uploading ~2.2 GB is faster and far safer than pulling 114 GB through Kaggle.

```bash
python scripts/prepare_upload.py --dry-run     # check the plan first
python scripts/prepare_upload.py               # ~25 min, resumable
```

Output: `C:/kaggle/upload/exp_data` — 217 classes, 18 crops, 84,123 images, ~2.2 GB at 288 px.
Everything trains and evaluates at 224 px, so storing more is ~4× the bytes for no benefit.

Then upload as a **private** dataset titled **`pde-sage-data`**:

```bash
kaggle datasets create -p "C:/kaggle/upload" -u
```

or Kaggle → Datasets → New Dataset → drag the folder.

---

## Run 2 — all VLM experiments

**Settings:** Accelerator **GPU T4×2** · Internet **ON** · Persistence **Files only**
**Add data:** `pde-sage-data`
**Secrets** (Add-ons → Secrets):

| secret | needed for |
|---|---|
| `GH_TOKEN` | cloning this private repo (fine-grained PAT, read-only Contents) |
| `LAVA_API_KEY` *or* `ANTHROPIC_API_KEY` | the ungrounded arm only — every other stage runs without it |

Paste **all of `kaggle/run2_experiments.py`** as one cell → Save Version → **Save & Run All
(Commit)** → close the tab. When it finishes: Output → Create Dataset → **`pde-results-2`**.

### What run 2 decides

The paper's headline — *"only source-grounded descriptors scale"* — is **retracted and not yet
rewritten**. `rich` is a keyword-retrieved bank in which, at scale C, only **8 of 51** held-out
classes get a unique descriptor: 17 fall back to the bare class name and **26 share text** with
another class. Beating that measures per-class *distinctness*, not grounding.

The ungrounded arm removes the confound — same model, same schema, same fall-through, temperature
1.0, three seeds; the only difference is that the "cite a retrievable source" constraint is dropped.

- `ungrounded ≈ grounded` → grounding is free and buys auditability. The cleaner paper.
- `ungrounded < grounded` → the sourcing constraint itself helps. A real finding.

Either is publishable. **No delta number goes in the manuscript until this lands.**

Run 2 also re-measures `rich` under the fixed matcher. `descriptors.text_for` normalised underscores
for the prompt but not for the match key, so all 13 multi-word bank entries (`powdery mildew`,
`citrus canker`, …) were unreachable for every label in the dataset. Every published `rich` accuracy
came from that broken matcher. New results are stamped `matcher_normalised`; anything unstamped is
set aside and recomputed rather than skipped.

---

## Run 3 — the 14 CNN baselines

**Settings:** GPU T4×2 · Internet ON · **Add data:** `pde-sage-data` *and* `pde-results-2`
**Secret:** `GH_TOKEN`

Paste **all of `kaggle/run3_cnns.py`** as one cell. Output → Create Dataset → **`pde-results-3`**.

Handles the two things that cost a previous sweep: CUDA OOM on a 14.56 GB T4 (batch is halved and
retried down to 16 rather than losing the architecture — three were lost this way and the cause was
misdiagnosed at the time as a missing timm arch; all three were `torch.OutOfMemoryError`), and
`--resume` never actually being passed, which restarted interrupted architectures from epoch 0 and
lost the EfficientNetV2-S run at 85.2 %.

---

## If a run stops early

Nothing is lost and nothing is redone. Every stage writes its own result file and skips work that
already exists; the probe cache resumes mid-encoder; CNNs resume from their last epoch checkpoint.

1. Let it commit (it stops itself before the wall).
2. Output → Create Dataset.
3. Attach that dataset to a fresh copy of the same notebook and run it again.

---

## After both runs

Download `results/*.json` into `docs/paper/`, then regenerate — **nothing is typed by hand**:

```bash
python docs/paper/make_tables.py --write
python docs/paper/make_tex_tables.py
python docs/paper/make_figures.py
python scripts/collect_results.py
python scripts/build_submission.py
```

Then §5.3, the abstract and the title get rewritten around what the ungrounded arm actually showed.

### Still open, independent of these runs

- **`PENDING-ZENODO-DOI`** in `main.tex` — COMPAG Option C needs the data deposited and linked.
  Deposit the `pde-sage-data` folder and cite SAGE at `bc9bd2899f`.
- The **181-URL verification** in `docs/paper/SOURCE_CHECKLIST.md`. Auditability is now the
  load-bearing claim, so this is critical path; only 16 of 217 records are page-verified.
