# Run logs — the reproducibility evidence trail

Raw stdout from every experiment, committed so that any number in `docs/paper/` can be traced
back to the run that produced it. 23 files, 99 KB total.

`*.log` is gitignored everywhere **except this folder** (`!logs/*.log` in `.gitignore`).

This index is generated — run `python scripts/index_logs.py` after adding a log, rather than
editing the table by hand.

| Log | Records | Feeds | Size |
|---|---|---|---|
| `cnn_efficientnetv2.log` | supervised CNN baseline: efficientnetv2 | `supervised_efficientnetv2.json`, `fig_cnn_training.png` | 0 KB |
| `cnn_mobilenetv3.log` | supervised CNN baseline: mobilenetv3 | `supervised_mobilenetv3.json`, `fig_cnn_training.png` | 1 KB |
| `cnn_mobilenetv4.log` | supervised CNN baseline: mobilenetv4 | `supervised_mobilenetv4.json`, `fig_cnn_training.png` | 1 KB |
| `cnn_resnet50.log` | supervised CNN baseline: resnet50 | `supervised_resnet50.json`, `fig_cnn_training.png` | 1 KB |
| `descriptor_audit.log` | descriptor spot-check before use | audit trail only | 7 KB |
| `descriptors_fill_all.log` | source-grounded descriptor generation | `descriptors/*.json` | 11 KB |
| `descriptors_fill_all_2.log` | source-grounded descriptor generation | `descriptors/*.json` | 10 KB |
| `edge_bench.log` | on-device latency and size | `edge_benchmark.json`, `fig_edge_pareto.png` | 2 KB |
| `edge_quant_bench.log` | INT8 quantisation sweep | `edge_quant_benchmark.json`, §5.7 | 3 KB |
| `eval_expA.log` | descriptor ablation, scale A | `zeroshot_eval_A.json`, `fig_descriptor_ablation.png` | 1 KB |
| `eval_expB.log` | descriptor ablation, scale B | `zeroshot_eval_B.json`, `fig_descriptor_ablation.png` | 1 KB |
| `eval_expC.log` | descriptor ablation, scale C | `zeroshot_eval_C.json`, `fig_descriptor_ablation.png` | 1 KB |
| `eval_expC_clean.log` | label-corrected zero-shot at scale C | `zeroshot_eval_C_clean.json` | 2 KB |
| `exp1_output.log` | EXP1 encoder bake-off | `run_all_bakeoff.json`, `fig_bakeoff.png` | 11 KB |
| `exp3_output.log` | EXP3 fine-tune + WiSE-FT | `run_all_exp3_lw11*.json`, `fig_wiseft.png` | 36 KB |
| `loco_full.log` | leave-one-crop-out with bootstrap CIs | `loco_s0_rich.json`, `tab_loco` | 2 KB |
| `metrics_expA.log` | abstention and top-5, scale A | `metrics_abstain_A.json`, `fig_riskcoverage.png` | 1 KB |
| `metrics_expB.log` | abstention and top-5, scale B | `metrics_abstain_B.json`, `fig_riskcoverage.png` | 1 KB |
| `metrics_expC.log` | abstention and top-5, scale C | `metrics_abstain_C.json`, `fig_riskcoverage.png` | 1 KB |
| `probe_all.log` | seen-crop probe, all tiers | `probe_seen_A/B/C.json`, `fig_seen_scaling.png` | 2 KB |
| `probe_all_b.log` | seen-crop probe, heavyweight tier | `probe_seen_*.json` | 2 KB |
| `probe_s1.log` | seen-crop probe, MobileCLIP-S1 | `probe_seen_*.json` | 1 KB |
| `supervised_output.log` | supervised CNN sweep, combined output | `supervised_*.json`, `tab_supervised` | 1 KB |

## Reading a log

Each begins with the resolved configuration (dataset revision, class counts, chance level) and
ends with the path of the JSON it wrote. When a manuscript number is questioned, the sequence is:
find the claim in `main.tex`, find the JSON the generator read, find the log that wrote it.

Logs from superseded builds are kept rather than deleted: a result that changed between builds is
evidence about the pipeline, and `docs/paper/_oldbuild_backup/` holds the corresponding JSONs.
