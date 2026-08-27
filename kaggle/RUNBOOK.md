# Kaggle runbook — one file, run it 2–3 times

Everything is in **`kaggle/RUN_THIS.py`**. Paste the whole file as one cell, run it, let it commit,
publish its Output as a Dataset, attach that to a fresh copy, run the *same file* again. It picks up
exactly where it stopped. Nothing is ever redone.

The previous single-cell attempt burned 12 hours and committed **nothing**. This file cannot do that:
it stops itself at `BUDGET_H = 8.5` and prints exactly what is left.

---

## Settings

| | |
|---|---|
| Accelerator | **GPU T4 × 2** (or **None/CPU** for a fetch-only first run — see below) |
| Internet | **ON** |
| Persistence | Files only |
| Add data | the previous run's output, every time after the first |

**Secrets** (Add-ons → Secrets):

| secret | needed for |
|---|---|
| `GH_TOKEN` | cloning this private repo — fine-grained PAT, read-only Contents |
| `HF_TOKEN` | the SAGE fetch — HuggingFace **read** token. Skip if you attach images. |
| `LAVA_API_KEY` **or** `ANTHROPIC_API_KEY` | the ungrounded arm **only**. Everything else runs without it. |

Then **Save Version → Save & Run All (Commit)** and close the tab.
When it finishes: **Output → Create Dataset**. Attach that next time.

---

## Getting the images in

Two options. The first is much faster.

**A. Build locally and upload (~25 min, recommended).** The images are already on this machine at
full resolution (60.8 GB, up to 3072 px):

```bash
python scripts/prepare_upload.py --dry-run     # check the plan
python scripts/prepare_upload.py               # 84,123 images, 1.7 GB at 288 px
```

Then upload `C:/kaggle/upload` as a **private** dataset titled `pde-sage-data` — kaggle.com →
Datasets → New Dataset → drag the folder. (The `kaggle` CLI also works, but needs
`~/.kaggle/kaggle.json` and the `id` edited in `dataset-metadata.json`; the web UI needs neither.)

Everything trains and evaluates at 224 px, so storing more is ~4× the bytes for nothing.

**B. Let Kaggle fetch it.** `RUN_THIS.py` does this automatically when no images are attached. It
pulls the pinned May release — **114 GB in ~10.7 GB shards**, so expect it to take most of a session
and possibly two. Run it on a **CPU session** first: the fetch never touches the GPU, and a CPU
session does not spend the 30 h/week GPU quota. The file detects this, fetches, and exits telling you
to switch to GPU.

---

## What runs, in order

Science first, the 8-hour CNN sweep last, so a truncated run still moves the paper forward.

1. data — attached, or fetched and verified (stops if the pull is short; a partial build would
   silently change every class count in the paper)
2. **ungrounded descriptors**, 3 seeds — text only, cheap, and it decides the paper
3. zero-shot, scales A/B/C
4. **the ungrounded control arm**, 3 seeds
5. abstention + top-5, A/B/C
6. label-corrected sensitivity run
7. seen-crop probe + leave-one-crop-out
8. the 14 supervised CNN baselines

With images attached that is roughly 15 h of compute, so **two runs**. Three if you also fetch.

---

## Why the last run died

It spent **11.8 of its 12 hours inside a single call** — a shard requested at t+548 s that never
returned a byte. Four independent causes, all fixed:

1. **A floating dataset revision.** `SHARD_REVISION` was `refs/convert/parquet`, an auto-generated
   branch HuggingFace **regenerated** when SAGE was reorganised on **2026-08-24**, three days before
   the run. The job was resuming a May-built `.shards_done.json` against August data. Now pinned to
   a commit SHA.
2. **No token.** The log carried *"You are sending unauthenticated requests to the HF Hub"*.
   Anonymous pulls are rate-limited, and a throttled HF connection does not fail — it stalls.
3. **No deadline.** `hf_hub_download` has no usable wall-clock bound and its backoff will retry a
   dead endpoint indefinitely. Shard downloads now run in a child process killed at 15 min and
   retried — a bad shard costs minutes, not a session.
4. **No budget.** A cell that overruns is killed and commits nothing, including finished stages.

### The part that matters for the paper, not just the run

SAGE shipped two incompatible releases:

| release | commit | shards | size | held-out crops | classes at C |
|---|---|---|---|---|---|
| **2026-05-07** | `bc9bd2899f` | 13 | 114 GB | **8** | **51** |
| 2026-08-24 | `dde0de8633` | 48 | 21 GB | 7 | 48 |

August is **not a superset**. Its own `canonical_mapping.json` marks all 14 Cotton entries
`"how": "no-canonical-crop"`, and the crop column of all 48 August shards contains **zero** Cotton
rows. Every published number here was measured with Cotton held out, so `config.py` pins **May**, and
`prepare_upload.py` reproduces the published split exactly:

```
scale A   3 held crops    16 classes
scale B   6 held crops    34 classes   (incl. Cotton ×3)
scale C   8 held crops    51 classes   (incl. Cotton ×3)
seen                     166 classes
```

Do not repoint the pin without re-measuring every zero-shot number. `PDE_SAGE_REVISION` overrides it
if you ever want August as a robustness check.

---

## What this run is actually for

The paper's headline — *"only source-grounded descriptors scale"* — is **retracted and not yet
rewritten**. `rich` is a keyword-retrieved bank in which, at scale C, only **8 of 51** held-out
classes get a unique descriptor: 17 fall back to the bare class name and **26 share text** with
another class. Beating that measures per-class *distinctness*, not grounding.

The ungrounded arm removes the confound — same model, same schema, same fall-through, temperature
1.0, three seeds; the only difference is that the "cite a retrievable source" constraint is dropped.

- `ungrounded ≈ grounded` → grounding is free and buys auditability. The cleaner paper.
- `ungrounded < grounded` → the sourcing constraint itself helps. A real finding.

Either is publishable. **No delta number goes in the manuscript until this lands**, and not from
fewer than three seeds.

The run also re-measures `rich`. `descriptors.text_for` normalised underscores for the prompt but not
for the match key, so all 13 multi-word bank entries (`powdery mildew`, `citrus canker`, …) were
unreachable for every label in the dataset. Every published `rich` accuracy came from that broken
matcher. Results are stamped `matcher_normalised`; anything unstamped is set aside and recomputed
rather than skipped.

---

## After it finishes

Download `results/*.json` into `docs/paper/`, then regenerate — **nothing in the paper is typed by
hand**:

```bash
python docs/paper/make_tables.py --write
python docs/paper/make_tex_tables.py
python docs/paper/make_figures.py
python scripts/collect_results.py
python scripts/build_submission.py
```

Then §5.3, the abstract and the title get rewritten around what the ungrounded arm showed.

### Open, and independent of these runs

- **`PENDING-ZENODO-DOI`** in `main.tex` — COMPAG Option C needs the data deposited and linked. Cite
  SAGE at `bc9bd2899f`, not "latest".
- The **181-URL pass** in `docs/paper/SOURCE_CHECKLIST.md`. Auditability is now the load-bearing
  claim, so this is critical path; only 16 of 217 records are page-verified.
