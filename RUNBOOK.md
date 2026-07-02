# RUNBOOK — run the whole pipeline on the RTX 4060

Everything below runs locally on your machine. Descriptor generation uses **Lava** (your $10 spend
key). Total descriptor cost with Sonnet ≈ **$1–2**, so the budget is safe.

Python interpreter used throughout (the pyenv one — the default `python` is a different venv):
```
/c/Users/PV Abhiram/.pyenv/pyenv-win/versions/3.11.9/python.exe
```
Below I call it `$PY`. In each terminal, first:
```bash
PY="/c/Users/PV Abhiram/.pyenv/pyenv-win/versions/3.11.9/python.exe"
cd /c/Projects/plant-disease-edge
```

---

## 0. One-time setup

```bash
# a) install deps (already done once; safe to re-run)
"$PY" -m pip install -r requirements.txt

# b) create your .env from the template, then paste your Lava key into it
cp .env.example .env
#   edit .env: set LAVA_API_KEY=... (leave LAVA_MODEL=anthropic/claude-sonnet-4-6)
#   PDE_DATASET_DIR and PDE_DATA_ROOT are already set to your existing data build.

# c) build the manifest (class list) from the existing images
"$PY" scripts/build_manifest.py --min-images 15
#   expect: heldout_crop=1,618 imgs (Coffee/Orange/Peach), train_crop crops listed.
```
`.env` is git-ignored — your key never gets committed.

---

## 1. Descriptors (Lava) + audit + VERIFY — DO THIS FIRST, it backs the core novelty

**Cost guard:** start with the 17 held-out diseases only (~$0.25). Verify quality, then do the rest.
```bash
# 1a) the 17 HEADLINE held-out descriptors first (cheap sanity check):
"$PY" scripts/build_descriptors.py --fill --classes-from heldout

# 1b) audit them: empty-check, ungrounded-field info, honest URL classification, sample to eyeball:
"$PY" scripts/audit_descriptors.py --sample 8 --check-urls

# 1c) OPTION A — apply web-verified verbatim citations over the LLM-recalled ones (headline diseases):
"$PY" scripts/apply_verified_citations.py     # sets "verified": true, fetchable sources, real quotes
"$PY" scripts/audit_descriptors.py --check-urls   # confirm: page-verified count up, 0 dead URLs

# 1d) fill ALL crops (seen + held, ~$1.30 total):
"$PY" scripts/build_descriptors.py --fill --classes-from all
#     (--limit 20 caps new calls per run if you want to spend in small steps)
```
Output: `descriptors/<Crop>.json`. **Why 1c matters:** Lava can't browse, so raw `source_url`/
`verbatim_quote` are *model-recalled* (some URLs dead, some quotes paraphrased). `apply_verified_citations.py`
holds citations we fetched from the real pages (UC-IPM / Wikipedia — authoritative and retrievable, since
APS blocks bots) and copied verbatim, stamping `"verified": true`. Only verified records back the
"source-grounded" claim; the rest are honestly "LLM-authored." `symptom_text` (what drives accuracy) is
solid either way. To add more verified diseases, extend the `V` dict in `apply_verified_citations.py`.

---

## 2. Descriptor ablation (bare vs crude vs rich vs grounded) — the core result

```bash
"$PY" scripts/evaluate.py --strategies bare crude rich grounded \
      --tiers lw11 lw21 lw35 --heavy --teachers
```
Runs your 4 models (S0/S1/S2/B) + SigLIP2 across all 4 descriptor strategies → `results/zeroshot_eval.json`.
`grounded` only beats `rich` once descriptors are filled+audited (stubs safely fall back to `rich`).

---

## 3. Top-5 + abstain / risk-coverage (makes 27% defensible)

```bash
"$PY" scripts/metrics.py --models s0 s1 s2 b --strategies rich grounded --reference
```
→ `results/metrics_abstain.json` (top-1, top-5, AURC, accuracy@coverage per model).

---

## 4. Rebuttal-proofing: LOCO + supervised baseline

```bash
# stability across crops (kills "cherry-picked held crops"):
"$PY" scripts/loco.py --model s0 --crops Coffee Orange Peach Apple Corn Potato --bootstrap 2000

# a conventional CNN baseline on seen crops (shows it CAN'T do unseen):
"$PY" scripts/supervised_baseline.py --arch mobilenetv3_small_100 --epochs 8 --batch 64
```
→ `results/loco_s0_rich.json`, `results/supervised_mobilenetv3_small_100.json`.

---

## 5. Edge / real-time benchmark (INT8 latency — the acceptance spine)

```bash
"$PY" -m pip install thop        # optional, for MACs
"$PY" scripts/benchmark_edge.py --models s0 s1 s2 b
```
→ `results/edge_benchmark.json` (params, MACs, FP32 + INT8 p50/p95 latency, ONNX sizes) and
`results/onnx/*.onnx`. **For the Raspberry Pi row:** copy the repo + `results/onnx/` to the Pi and run
`"$PY" scripts/benchmark_edge.py --models s0 --no-onnx` there.

---

## 6. Figures + commit

```bash
"$PY" docs/paper/make_figures.py          # regenerates all figures incl. fig_wiseft.png

# copy the result JSONs into the repo for reproducibility, then commit:
cp /c/kaggle/working/results/*.json docs/paper/ 2>/dev/null || true
git add scripts/ docs/paper/ *.md .env.example requirements.txt
git commit -m "Descriptor(Lava)+ablation, abstain metrics, LOCO, baseline, edge benchmark; sync paper"
```
Do NOT commit `.env`, `*.pt` weights, or the raw dataset (all git-ignored).

---

## Recommended order
`0 → 1 → 2 → 3 → 5 → 4 → 6`. The only paid step is **1** (Lava, ~$1–2). Everything else is free and
runs on the 4060/CPU. Step **5** (edge latency) is what turns this into a publishable *system* paper —
don't skip it.
