# Run logs — reproducibility evidence trail

Raw stdout from every experiment run, kept in git (small, ~73 KB total) so results in
`docs/paper/` can be traced back to the run that produced them.

`*.log` is gitignored everywhere **except this folder** (`!logs/*.log` in `.gitignore`).

| Log | Produced by | Feeds |
|---|---|---|
| `exp1_output.log` | EXP1 encoder bake-off | `docs/paper/run_all_bakeoff.json`, `fig_bakeoff.png` |
| `exp3_output.log` | EXP3 fine-tune + WiSE-FT | `run_all_exp3_lw11*.json`, `fig_wiseft.png` |
| `eval_expA/B/C.log` | descriptor ablation (16 / 34 / 51 classes) | `fig_descriptor_ablation.png` |
| `metrics_expA/B/C.log` | top-5 + abstain metrics | `metrics_abstain.json`, `fig_riskcoverage.png` |
| `loco_full.log` | LOCO leave-one-crop-out (18 crops) | `loco_s0_rich.json` |
| `cnn_*.log` | supervised CNN baselines | `supervised_mobilenetv3_small_100.json` |
| `supervised_output.log` | supervised baseline driver | same |
| `edge_bench.log` | edge latency benchmark | `edge_benchmark.json` ⚠️ *see note* |
| `descriptors_fill_all.log` | descriptor generation (Sonnet/Opus) | `descriptors/*.json` |

> ⚠️ **`edge_bench.log` / `edge_benchmark.json` are known-bad for INT8** — the INT8 rows show a
> 21–23× slowdown caused by `quantize_dynamic()` defaults, not by the models. Superseded by
> `kaggle/benchmark_quantization.py`. Do not quote the INT8 numbers from that run.

**Convention:** when a run supersedes an old one, keep both logs and note it here — don't overwrite.
