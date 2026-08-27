# scripts/ — the validated pipeline

The method that survived Phase-0 de-risk (7 experiments; see `docs/paper/findings_log.md`):
**frozen pretrained compact CLIPs + descriptor text prototypes** for cross-crop zero-shot disease
diagnosis. Training a tiny model to *learn* the alignment (from scratch or by specializing on seen
crops) was shown **not** to help unseen-crop zero-shot — so we do not train backbones.

## Modules
| File | Job |
|---|---|
| `config.py` | single source of truth — crops, aliases, paths, the model family (`MODEL_TIERS`), teachers, shard-fetch + eval settings |
| `sage_data.py` | download + filter SAGE parquet shards → `<DATASET_DIR>/<Crop>___<Disease>/*.jpg` (incremental, resumable, dedup) |
| `descriptors.py` | `bare` / `crude` / `rich` / `grounded` text prototypes; `build_prototypes(...)`; rich symptom bank + loader for source-grounded `descriptors/<crop>.json` |
| `zeroshot.py` | frozen-model eval core: load model, embed images, nearest-prototype accuracy |
| `evaluate.py` | **driver** — fetch data → model-family × strategy sweep → `<RESULTS_DIR>/zeroshot_eval.json` |
| `build_descriptors.py` | Phase A2: generate source-grounded `{value, source_url, verbatim_quote}` descriptors (Claude or stubs) |

## Run it (Kaggle: GPU + Internet ON, repo cloned)
```bash
PDE_DATA_ROOT=/kaggle/working python scripts/evaluate.py             # 11/21/35M tiers, bare/crude/rich
PDE_DATA_ROOT=/kaggle/working python scripts/evaluate.py --teachers  # + SigLIP2 reference
python docs/paper/make_figures.py                                    # regenerate figures from results
```
`evaluate.py` self-installs deps and fetches held-out images on first run.

## The data tip (do this once for Phase A)
SAGE crops are scattered across 13 large parquet shards (Coffee is rare in the early ones). Build the
subset **once**, **save `DATASET_DIR` as a Kaggle Dataset**, and **attach** it next time — every run
then skips downloading and gets all crops (incl. Coffee) with a clean taxonomy.

## Result → paper
`zeroshot_eval.json` feeds `docs/paper/make_figures.py` and the tables in `docs/paper/TABLES.md`.

Validated family: **MobileCLIP2-S0 ~11M / MobileCLIP-S1 ~21M / MobileCLIP2-S2 ~35M** (image-encoder
params; only the image encoder deploys). A ~5M tier is not off-the-shelf — it would need TinyCLIP-style
weight-inherited distillation (optional stretch).
