# KAGGLE RUNBOOK — run the whole GPU pipeline for free

Everything still outstanding runs on Kaggle's free tier. No credit card, no Vast.

> **The one rule that makes this work:** build the dataset **once**, save it as a **Kaggle Dataset**,
> and *attach* it to every later notebook. Kaggle gives you only **~20 GB** in `/kaggle/working` and a
> SAGE parquet shard is **~10 GB** — you cannot re-fetch data every session.

---

## 0. Kaggle limits you must design around

| Limit | Value | Consequence |
|---|---|---|
| GPU quota | **30 h/week** (T4×2 or P100) | plenty — the whole plan is ~8–10 h |
| Session length | **9 h** GPU / 12 h CPU | every stage below fits in one session |
| `/kaggle/working` | **~20 GB**, persists per version | too small to hoard shards → save a Dataset |
| `/kaggle/temp` | ~73 GB, wiped at session end | fine as scratch during the fetch |
| Internet | **off by default** | must switch ON (needs phone verification) |
| Idle timeout | 40 min interactive | **always use Save Version → Run All** |

**Notebook settings for every session:** `Accelerator = GPU T4 x2` (or P100) · `Internet = ON` ·
`Persistence = Files only`.

**Run everything as a committed batch job:** *Save Version → **Save & Run All (Commit)***. It survives
a closed tab and the 40-minute idle killer. Interactive mode is only for quick checks.

---

## SESSION 1 — build the data Dataset (do this once, ~2–4 h)

This is the expensive one. Everything afterwards reuses it.

New Notebook → GPU (not needed but harmless) → Internet ON → paste:

```python
!git clone https://YOUR_TOKEN@github.com/Abhiram970/plant-disease-edge.git /kaggle/working/pde
%cd /kaggle/working/pde
!pip install -q --no-deps timm open_clip_torch
!pip install -q safetensors pyyaml huggingface_hub pyarrow tqdm regex ftfy

import os
os.environ["PDE_DATA_ROOT"]   = "/kaggle/working"
os.environ["PDE_DATASET_DIR"] = "/kaggle/working/exp_data"

# incremental + resumable: keeps .shards_done.json, only pulls missing shards
!python scripts/sage_data.py --role all
!python scripts/build_manifest.py --min-images 25
!du -sh /kaggle/working/exp_data && ls /kaggle/working/exp_data | wc -l
```

**Then publish it:** notebook right pane → **Output → Create Dataset** → name it
**`pde-sage-data`**. That snapshot (images + `manifest.csv`) is what every later session attaches.

> Repo is private, so `YOUR_TOKEN` = a fine-grained GitHub PAT with read-only Contents.
> Better: put it in **Add-ons → Secrets** as `GH_TOKEN` and read it with
> `from kaggle_secrets import UserSecretsClient; tok = UserSecretsClient().get_secret("GH_TOKEN")`
> so it never appears in a saved notebook.

---

## The standard header (paste at the top of Sessions 2–5)

Attach **`pde-sage-data`** via *Add data* first, then:

```python
!git clone https://YOUR_TOKEN@github.com/Abhiram970/plant-disease-edge.git /kaggle/working/pde
%cd /kaggle/working/pde
!pip install -q --no-deps timm open_clip_torch
!pip install -q safetensors pyyaml huggingface_hub pyarrow tqdm regex ftfy

import os
os.environ["PDE_DATA_ROOT"]   = "/kaggle/working"                      # results land here
os.environ["PDE_DATASET_DIR"] = "/kaggle/input/pde-sage-data/exp_data"  # READ-ONLY attached images
!cp /kaggle/input/pde-sage-data/manifest.csv /kaggle/working/ 2>/dev/null
import torch; print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

No download. Straight to compute.

---

## SESSION 2 — EXP3 WiSE-FT sweep ⭐ P0, blocks the paper (~1.5–2 h)

The α-table in the paper is from the old small config and the full-data run died at epoch 1/5.

```python
!python kaggle/run_all_win.py --only exp3 --tier lw11 --ft-epochs 5 --batch 64
!ls -la exp_out/
```
→ `exp_out/run_all_exp3_lw11.json`. **Download it** (Output pane) before the session ends.

If you hit CUDA OOM: `--batch 32`.

---

## SESSION 3 — zero-shot scale study A/B/C ⭐ P0 (~1.5–2 h)

This is what kills the 27 %-vs-17 % inconsistency: all three configs from one consistent sweep.

```python
for exp in ["A", "B", "C"]:
    !python scripts/evaluate.py --exp {exp} --strategies bare crude rich grounded \
            --tiers lw11 lw21 lw35 --heavy --teachers
    !cp /kaggle/working/results/zeroshot_eval.json /kaggle/working/zeroshot_eval_exp{exp}.json
!ls -la /kaggle/working/*.json
```

Download all three `zeroshot_eval_exp*.json`.

---

## SESSION 4 — seen-head probe A/B/C (~1.5–2 h)

The "seen accuracy vs #classes" half of the scale study.

```python
for exp in ["A", "B", "C"]:
    !python scripts/probe_seen.py --exp {exp} --epochs 40
    !cp /kaggle/working/results/probe_seen.json /kaggle/working/probe_seen_exp{exp}.json
```

---

## SESSION 5 — CNN baselines + metrics + LOCO (~2.5–3 h)

Resumable — the runner **skips any arch whose result JSON already exists**, so if the 9 h clock runs
out, just re-run this session and it continues where it stopped.

```python
!python scripts/run_cnn_baselines.py \
     --archs resnet50 mobilenetv3_large_100 mobilenetv4_conv_small \
             efficientnet_b0 convnextv2_nano fastvit_t8 \
     --epochs 12 --batch 128 --workers 4

!python scripts/metrics.py --models s0 s1 s2 b --strategies rich grounded --reference
!python scripts/loco.py --model s0 --strategy rich --bootstrap 2000
!ls -la /kaggle/working/results/
```

> `--workers 4` (not 8): Kaggle VMs have ~4 vCPU; more workers just thrash.

---

## Bring the results home

Per session: right pane → **Output** → download the JSONs. Then locally:

```bash
cp ~/Downloads/*.json docs/paper/
"$PY" docs/paper/make_figures.py
git add docs/paper/ && git commit -m "Kaggle run: EXP3 sweep, A/B/C scale study, CNN baselines" && git push
```

---

## Priority if the week's 30 h runs short

| Order | Session | Why |
|---|---|---|
| 1 | **1 (data)** | nothing runs without it |
| 2 | **2 (EXP3)** | **P0 — blocks the paper** |
| 3 | **3 (scale study)** | **P0 — fixes the inconsistent numbers** |
| 4 | 5 (CNN + metrics + LOCO) | strengthens the baseline table |
| 5 | 4 (probe A/B/C) | nice-to-have scaling curve |

Sessions 2 and 3 alone are enough to finish the draft.

---

## Gotchas

- **Internet OFF** → `git clone` and HuggingFace both fail with confusing errors. Check the sidebar toggle.
- **Not committed** → interactive sessions die at 40 min idle. Use *Save & Run All (Commit)*.
- **`/kaggle/working` full** → you re-fetched shards instead of attaching `pde-sage-data`. Check
  `PDE_DATASET_DIR` points at `/kaggle/input/...`.
- **Attached input is read-only** → never point `PDE_DATA_ROOT` at `/kaggle/input`; results must go to
  `/kaggle/working`.
- **Torch reinstall** → always `pip install --no-deps timm open_clip_torch`; a plain install drags in a
  2 GB torch wheel and can break CUDA.
- **Results vanished** → they were in `/kaggle/working` of a session you didn't commit. Commit, or
  download before closing.
- **`exp_out/` vs `results/`** → `kaggle/run_all_win.py` writes to `./exp_out/`; the `scripts/*` tools
  write to `$PDE_DATA_ROOT/results/`. Check both when collecting.

---

## What Kaggle can NOT do

- **Raspberry Pi / ARM latency row** — needs real hardware. Still the most likely reviewer attack on an
  "edge" paper.
- **The 18 dead source URLs** — manual, per-URL web work.
- **Elsevier LaTeX conversion, authors, funding** — writing.
