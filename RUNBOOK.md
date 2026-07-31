# RUNBOOK — run every piece of this project

Two execution surfaces:

| Surface | What runs there | Cost |
|---|---|---|
| **Local** (your RTX 4060 laptop) | descriptors, zero-shot eval, metrics, LOCO, edge benchmark, figures | free (+ ~$1–2 LLM API) |
| **Vast.ai** (rented GPU) | EXP3 WiSE-FT sweep, A/B/C scale study, CNN baseline family | ~$1.65 of a $4.60 budget |

> Anything that **trains on 56k+ images** goes to Vast. Everything else runs locally.
> Cloud plan lives in [`vast/README.md`](vast/README.md).

---

## 0. TL;DR — the order that matters

```bash
#  LOCAL, in this order
1. setup            (§1)   once
2. build_manifest   (§2)   ~1 min
3. descriptors      (§3)   ~$1-2, the core novelty
4. zero-shot eval   (§4)   the headline table
5. metrics/abstain  (§5)   makes the modest top-1 defensible
6. LOCO             (§6)   anti-cherry-pick
7. edge benchmark   (§7)   DONE — re-run only if hardware changes
8. figures + commit (§8)

#  CLOUD (vast/README.md) — the GPU-bound remainder
9. EXP3 WiSE-FT sweep, A/B/C scale study, CNN baselines
```

---

## 1. Setup (once)

The default `python` on this machine is a different venv. **Always use the pyenv interpreter:**

```bash
PY="/c/Users/PV Abhiram/.pyenv/pyenv-win/versions/3.11.9/python.exe"
cd /c/Projects/plant-disease-edge

"$PY" -m pip install -r requirements.txt
```

`.env` (git-ignored) must contain:
```
LAVA_API_KEY=...            # LLM spend key for descriptors
LAVA_BASE_URL=https://api.lava.so/v1
LAVA_MODEL=claude-sonnet-4-6
PDE_DATASET_DIR=C:\kaggle\working\exp_data    # existing image build (552 class folders)
PDE_DATA_ROOT=C:\kaggle\working
```

Verify:
```bash
"$PY" -c "import torch, open_clip, onnxruntime; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

---

## 2. Manifest (class list from the images on disk)

```bash
"$PY" scripts/build_manifest.py --min-images 25
```
→ `data/manifest.csv`. Re-run whenever the image set changes.

---

## 3. Descriptors — the core novelty ⚠️ costs money

Descriptors are what make zero-shot possible; **quality is the paper's main lever** (+8–15 pp).

```bash
# 3a) held-out crops first (cheap sanity check, ~$0.25)
"$PY" scripts/build_descriptors.py --fill --classes-from heldout

# 3b) audit: coverage, empty fields, URL health
"$PY" scripts/audit_descriptors.py --sample 8 --check-urls

# 3c) apply hand-verified verbatim citations over model-recalled ones
"$PY" scripts/apply_verified_citations.py
"$PY" scripts/audit_descriptors.py --check-urls      # confirm verified count up

# 3d) everything (~$1.30)
"$PY" scripts/build_descriptors.py --fill --classes-from all
#     --limit 20   caps new API calls per run if you want to spend in steps
```

**Expected behaviour, not a bug:** the grounding prompt forbids uncited claims, so for obscure or
mislabelled classes the model returns *empty fields and says so* rather than fabricating. Those stay
`status: stub`. Re-running will not fix them — they need a **real source** fed in (extend the `V` dict
in `apply_verified_citations.py`).

**Known unfixable-by-LLM classes** (SAGE label noise — document, don't fill):
- `Wheat/Resistance_Phenotype`, `..._Moderately_Resistant`, `..._Moderately_Susceptible` — these are
  *phenotype ratings, not diseases*. Recommend excluding them from the eval and saying so.
- `Coffee/Miner` — an **insect pest** (*Leucoptera coffeella*), not a pathogen.
- `Coffee/Cerscospora` — misspelling of *Cercospora coffeicola* (brown eye spot).

---

## 4. Zero-shot descriptor ablation — the headline table

```bash
"$PY" scripts/evaluate.py --strategies bare crude rich grounded \
      --tiers lw11 lw21 lw35 --heavy --teachers
```
→ `results/zeroshot_eval.json`. Add `--exp A|B|C` to run a specific scale-study configuration
(A = 4 seen/3 held, B = 8/6, C = 10/8). **The A/B/C sweep is a Vast stage** (§9) — locally, run one.

---

## 5. Top-5 + abstain / risk-coverage

```bash
"$PY" scripts/metrics.py --models s0 s1 s2 b --strategies rich grounded --reference
```
→ `results/metrics_abstain.json` (top-1, top-5, AURC, accuracy@coverage).
This is what makes a ~27 % top-1 defensible to a reviewer.

---

## 6. LOCO (anti-cherry-pick)

```bash
"$PY" scripts/loco.py --model s0 --strategy rich --bootstrap 2000
```
→ `results/loco_s0_rich.json`. Shows held-out crops sit mid-range, not hand-picked.

---

## 7. Edge / quantisation benchmark ✅ DONE (27 Jul)

```bash
"$PY" kaggle/benchmark_quantization.py                       # all 4 models, ~15 min
"$PY" kaggle/benchmark_quantization.py --models s0 --runs 20 # quick check
```
→ `results/edge_quant_benchmark.{json,md}` (already copied into `docs/paper/`).

Measures torch FP32 / ONNX FP32 / FP16 / INT8-dynamic / INT8-static(QDQ) + a per-graph diagnosis.

> ⚠️ The **older** `scripts/benchmark_edge.py` produced the *superseded* INT8 numbers (21–23× slowdown
> from `quantize_dynamic` defaults). Don't quote `docs/paper/edge_benchmark.json`.

**Still missing:** the **Raspberry Pi / ARM row** — copy the repo + `results/onnx/` to a Pi and run the
same script there. No rented GPU can substitute for this.

---

## 8. Figures + commit

```bash
"$PY" docs/paper/make_figures.py         # regenerates all 11 figures
cp results/*.json docs/paper/            # bring results into the repo
git add scripts/ docs/paper/ descriptors/ logs/ *.md
git commit -m "..."   &&  git push
```

Never commit: `.env`, weights, `results/` (ONNX exports), raw dataset — all git-ignored.

---

## 9. GPU work → **Kaggle (free)**

Everything GPU-bound runs on Kaggle's free 30 h/week tier.
**Full guide: [`kaggle/RUNBOOK_KAGGLE.md`](kaggle/RUNBOOK_KAGGLE.md).**

The rule that makes it work: **build the dataset once, save it as a Kaggle Dataset
(`pde-sage-data`), attach it to every later notebook.** `/kaggle/working` is only ~20 GB and a SAGE
shard is ~10 GB, so you cannot re-fetch data each session.

| Session | What | Priority |
|---|---|---|
| 1 | SAGE data → save as Kaggle Dataset | mandatory, once |
| 2 | **EXP3 WiSE-FT full sweep** | **P0 — blocks the paper** |
| 3 | **Zero-shot scale study A/B/C** | **P0 — resolves the 27 %-vs-17 % mismatch** |
| 4 | Seen-head probe A/B/C | P1 |
| 5 | CNN baselines + metrics + LOCO | P1 |

Sessions 2 + 3 alone are enough to finish the draft. Always use **Save Version → Save & Run All
(Commit)** so the 40-minute idle timeout can't kill a run.

*(A paid Vast.ai variant of the same plan is kept in [`vast/README.md`](vast/README.md) if you ever
want a single uninterrupted box instead of 9-hour sessions.)*

---

## 10. Status — what's done vs open

| Item | State |
|---|---|
| Descriptor pipeline + audit + verified citations | ✅ built |
| Descriptor coverage | ✅ 156/217 filled · 16 page-verified · **headline held-out crops 16/16 complete** (Coffee 5/5, all page-verified). 4 remaining stubs are Wheat label artefacts, not gaps |
| Zero-shot ablation (bare/crude/rich/grounded) | ✅ |
| Encoder bake-off (6 encoders) | ✅ |
| Hybrid (trained seen + zero-shot unseen) | ✅ |
| Top-5 + abstain / risk-coverage | ✅ |
| LOCO | ✅ |
| Supervised CNN baselines | ⚠️ 1 of 6+ done → **Vast stage 5** |
| EXP3 WiSE-FT full-data sweep | ❌ died at epoch 1/5 → **Vast stage 2** |
| Scale study A/B/C | ❌ → **Vast stage 3** |
| Edge/quantisation benchmark | ✅ redone 27 Jul, INT8 root-caused |
| Raspberry Pi / ARM latency row | ❌ needs real hardware |
| Paper draft | ✅ 254 lines, §5.9–5.10 updated with corrected numbers |
| Elsevier LaTeX conversion, authors, funding | ❌ |
| Multi-seed CIs on headline table | ❌ |

---

## 11. Troubleshooting

- **`python` has no torch** → you used the wrong interpreter; use `$PY` from §1.
- **Descriptor returns empty fields** → expected for ungroundable/mislabelled classes (§3). Feed a real
  source rather than loosening the prompt; the refusal *is* the anti-hallucination guarantee.
- **CUDA OOM** → lower `--batch` (32 for fine-tuning, 64 for CNN baselines).
- **`open_clip` missing on a rented box** → re-run `bash setup_vast.sh` (installs it `--no-deps`).
- **`$'\r': command not found` on Linux** → CRLF checkout; `.gitattributes` forces LF for `*.sh`, so
  re-clone rather than editing on Windows and copying.
- **Figures look stale** → they read the JSONs in `docs/paper/`; copy new results there first.
