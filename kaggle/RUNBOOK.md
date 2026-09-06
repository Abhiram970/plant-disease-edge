# Kaggle runbook

Paste **`kaggle/LAUNCH.py`** as one cell, set `PART`, run. It clones the pinned branch and
executes the chosen stage — there is nothing else to copy, and no risk of pasting a file that
only prints itself.

```python
!git clone -q --depth 1 --branch paper/draft-audit-2026-09-01 \
  https://github.com/Abhiram970/plant-disease-edge.git /tmp/pde
%run /tmp/pde/kaggle/LAUNCH.py
```

| `PART` | what it runs | approx. | API key |
|---|---|---|---|
| **`"fixup"`** | **the two stages the morning run lost: WiSE-FT, then the 14 CNNs** | **6.5 h** | **no** |
| `"tonight"` | descriptors, zero-shot A/B/C, control arms, probe, abstention | 3.9 h | required |
| `"morning"` | extra seeds, paired comparison, remaining tables, 14 CNNs | 7.8 h | required |
| `"1"` / `"2"` / `"3"` | the same work split into single-purpose stages | — | part 1 only |

**Run `"fixup"` next.** The 2026-09-06 morning session (10.12 h, exit 0) completed everything
except those two stages, and its results carry forward from the attached output rather than being
recomputed. Attach `pde-sage-data` **and** that session's output, then run with `PART = "fixup"`.

Every stage is resumable. A run that reaches its budget stops cleanly, prints what is left, and
a re-run of the same cell continues from there — finished work is skipped, never redone. When a
stage completes, publish the notebook Output as a Dataset and attach it to the next run so
results and descriptor text carry forward.

The stages are generated from one shared bootstrap (`_pde_common.py`) by `build_parts.py`,
`build_tonight.py` and `build_morning.py`, so a fix lands in every runner at once. Edit the
generators, not the generated `RUN_*.py` files.

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

**B. Let Kaggle fetch it.** The runner does this automatically when no images are attached. It
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

## Why the morning run finished 12 of 14 stages

The 2026-09-06 session exited 0 after 10.12 h and produced the paired Section 5.3 comparison, both
control arms at 7-8 usable seeds, zero-shot A/B/C, the clean-label sensitivity run, coverage, the
seen-crop probe, LOCO and abstention. Two stages did not survive, for unrelated reasons.

**The 14 CNNs finished 0 of 14.** Ten architectures died within 25 s each:

```
RuntimeError: GET was unable to find an engine to execute this computation
```

raised from `F.conv2d` inside a depthwise convolution. Every one of the ten is a depthwise design
(MobileNetV3/V4, EfficientNet, FastViT, ConvNeXt-V2, EfficientNetV2); the four that ran are
plain-conv nets. The cause is `torch.cuda.is_bf16_supported()`, which returns `True` on the T4:
Turing has no native bf16, and cuDNN ships no depthwise convolution engine for the emulated path.
Precision is now selected from `torch.cuda.get_device_capability()` — fp16 on the T4, bf16 only on
sm_80 and later — with an automatic step-down ladder (bf16 → fp16 → fp16 contiguous → fp32) if a
kernel is still missing.

The batch probe made this worse rather than catching it. It printed the same engine error and then
returned the requested batch anyway, because it only reduced on `"OOM" in out`. The probe now walks
the same ladder as the trainer and reports which precision worked; a probe that finds no workable
configuration returns `None` and the architecture is skipped and named in the receipt, instead of
spending its whole budget failing.

The other four (densenet121, regnety_040, resnet50, resnet101) trained correctly and were killed by
the 0.75 h per-architecture cap partway through **epoch 2** — densenet121 reached 81.6 % and
resnet50 79.8 % on epoch 1, inside the cap. Worse, the cleanup deleted each checkpoint
unconditionally, so a completed epoch was thrown away every time. The cap is now 1.6 h, and a
checkpoint is deleted only once the architecture has produced a readable JSON, so an overrun
resumes.

**WiSE-FT completed but its sweep was unusable.** Fine-tuning converged (loss 4.463 → 3.179 →
2.400), the encoder moved (relative L2 = 0.314) and α=0 reproduced the frozen probe (58.7 % vs
58.8 %) — so the three gates that had caught earlier bugs all passed. But seen accuracy went
58.7 → **45.1** → 63.8 while unseen fell 21.6 → 7.8 → 1.8. A midpoint below *both* endpoints means
the frozen and fine-tuned weights are not linearly mode-connected, and interpolating between them
is meaningless.

The cause was the head: `nn.Linear` was randomly initialised and trained jointly with the unfrozen
encoder, so its large early gradients pushed the encoder out of the pretrained basin before it had
learned anything. The head is now fitted on frozen features first and used to warm-start
fine-tuning — that fit is the α=0 reference the sweep needs anyway, so it costs no extra pass — the
encoder is excluded from weight decay (decaying toward zero pulls it away from the pretrained
weights, the opposite of what WiSE-FT wants), and the sweep runs five alphas instead of three so a
non-monotonicity can be distinguished from one noisy point.

The superseded sweep is kept as `wiseft_SUPERSEDED_2026-09-06.json`, and the fix-up runner moves any
carried-forward `wiseft.json` aside before re-running, so a failed re-run cannot leave the broken
numbers in place for the table generator to pick up.

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
