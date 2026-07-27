# Phase-0 de-risk — findings log (evidence trail)

Held-out eval = Coffee/Orange/Peach, 17 classes, 1,618 imgs, chance **5.9%**. Crude keyword
descriptors unless noted. Student backbone for distill runs = `edgenext_small` (~5.5M).

| # | Experiment | Setup | Held-out zero-shot | Verdict |
|---|---|---|---|---|
| 1 | Spike (invalid) | bare prompts, held collapsed to 1 crop (Coffee) | student 32% on 5 Coffee classes | VOID (single-crop, teacher ~chance) |
| 2 | Distill from scratch | CLIP-B/16 teacher, 2 crops, mimic loss | student 11.6% (54% retention) | NO-GO (recipe) |
| 3 | + text anchoring | CLIP-B/16, 3 train crops | student 11.0% | no change |
| 4 | Fair-shot distill | **SigLIP2** teacher, 4 train crops, anchoring | **student 11.0%, teacher 25.6%** | NO-GO — stronger teacher didn't help ⇒ **architecture** |
| 5 | Probe pretrained CLIPs | frozen, no training | **MobileCLIP2-S0 (11M) 19.9%**, flat 11–300M @17–22%, SigLIP2 25.6% | **FORK A: feasible** ⇒ pivot to pretrained CLIPs |
| 6 | Specialize (adapter) | MobileCLIP2-S0 + residual adapter + SigLIP2 KD | base 19.9% → **15.4% (−4.5pp)** | NO LIFT — catastrophic forgetting |
| 7 | Descriptor quality | bare/crude/**rich** on frozen models, 8 classes (Orange+Peach), chance 12.5% | **rich best on all**: S0 29.5→**37.5%** (+8), S1 24.8→**39.8%** (+15), SigLIP2 34.8→**49.5%** (+14.7); Orange/HLB 8.9→**95.6%** on 11M | **POSITIVE — descriptor lever validated** |
| 8 | Encoder bake-off | rich zero-shot, **3 held crops, 17 classes**, chance 5.9% | **SigLIP2 31.5%** (best) > S2 28.7% > S0 26.9% > S1 22.5% > BioCLIP2 9.6% (poor, drop). SCOLD loaded (class LVL) but **5.3% — adapter issue, not valid** (RoBERTa from base; preprocess mismatch) | SigLIP2=teacher; lightweight ~22-29% (noisy ranking); drop BioCLIP2; SCOLD needs inference.py |
| 9 | Train seen / keep unseen | MobileCLIP2-S0 11M, linear probe, 80 seen classes, 3 held crops | **SEEN trained 67.0% vs zero-shot 8.5%** (8× win); **UNSEEN zero-shot 26.9% preserved** | **HYBRID VALIDATED** — train seen, zero-shot unseen, one frozen 11M model |
| 10 | Fine-tune + WiSE-FT (EXP3) | MobileCLIP2-S0, ft 5 epochs, alpha sweep | theta0 ALIASING BUG (deepcopy) made alpha=0 = forgotten model (15.9% not 26.9%); **FIXED** (reload pristine weights); clean sweep re-run pending | re-run `--only exp3` |

## What is established
- **Works:** frozen compact pretrained CLIP + descriptors does cross-crop zero-shot (~20% @ 11M, 3.4× chance, ~78% of 93M SigLIP2).
- **Works:** accuracy flat 11M→300M ⇒ size is not the bottleneck (efficiency pillar).
- **Does NOT work:** training a small model to *learn/improve* the alignment — from scratch (≈chance) or specialize-on-seen (−4.5pp, forgetting). Literature-consistent (WiSE-FT).
- **Untested lever:** descriptor quality (crude stubs everywhere so far; SAGE says source-grounded +14–16pp).

## Edge / quantization (run 11 — 27 Jul 2026, local CPU, ORT 1.26, 50 runs, batch 1, 224px)

Re-ran the edge benchmark because the earlier `edge_benchmark.json` reported INT8 as **21–23×
slower** than FP32. **Reproduced and root-caused it.** Cause: `quantize_dynamic()` with defaults
(no `quant_pre_process`, no `per_channel`, no calibration). Full table:
`docs/paper/edge_quant_benchmark.{json,md}`.

| model | params | ONNX FP32 | INT8 dynamic (old) | INT8 static QDQ (fixed) | best config |
|---|---|---|---|---|---|
| MobileCLIP2-S0 | 11.4M | **17.35 ms** / 45.8 MB | 320.0 ms (18.4× slower) | 62.2 ms (3.6× slower) | **FP32** |
| MobileCLIP-S1 | 21.5M | **33.74 ms** / 86.5 MB | 614.9 ms (18.2× slower) | 79.5 ms (2.4× slower) | **FP32** |
| MobileCLIP2-S2 | 35.8M | **49.19 ms** / 143.6 MB | 923.8 ms (18.8× slower) | 98.7 ms (2.0× slower) | **FP32** |
| MobileCLIP-B | 86.4M | 100.78 ms / 345.6 MB | 60.4 ms (1.7× faster) | **46.39 ms** / 87.4 MB (2.2× faster) | **INT8 static** |

**The split is architectural and perfectly consistent.** The three lightweight models are
FastViT-style **hybrids with depthwise convolutions**; MobileCLIP-B is a **pure ViT**. Mechanistic
evidence from the node histograms — `float conv left` (convs ORT could *not* quantize):

| model | float convs left after static quant | speedup |
|---|---|---|
| S0 / S1 / S2 | **119 / 215 / 235** | 0.28× / 0.42× / 0.50× |
| B | **3** | 2.17× |

So on the hybrids the graph must dequantize → float-conv → requantize hundreds of times per
inference (1290–2536 convert nodes). Dynamic quant instead converts every conv to `ConvInteger`
(0 float convs left) which is pathologically slow for depthwise — hence the ~18× cliff.

**Established:**
- **INT8 is NOT viable for hybrid conv-transformer VLMs on the ORT CPU EP.** Static QDQ recovers
  5–9× over the dynamic path but still never beats FP32 on S0/S1/S2.
- **Deployment config for the lightweight tier is ONNX FP32.** S0 = **17.35 ms ≈ 58 FPS** on CPU at
  11.4M params. (Faster than the old 21.5 ms because `ORT_ENABLE_ALL` is now on.)
- **INT8 is a *size* lever, not a speed lever, for the hybrids** — 3.5× smaller (45.8 → 12.9 MB)
  at 3.6× the latency. Use only when memory-bound.
- **FP16 fails on the ORT CPU EP for all four** (no fp16 CPU kernels) — it is a GPU/NPU option only.
- Caveat: verdicts are latency-measured, not node-counted (ORT fuses QDQ at session-init, so a
  serialized-graph scan is misleading). Calibration used random tensors — valid for latency; an
  INT8 *accuracy* claim needs `--calib-dir` with real crop images.

## Implications for the paper
- Headline = **descriptor-driven cross-crop zero-shot on FROZEN compact edge VLMs + efficiency Pareto + abstain**; specialization demoted to a seen-crop booster.
- Must report **top-5 / abstain-gated** accuracy (fine-grained 17-class top-1 is modest).
- Contribution must be *built by us* (descriptor pipeline + family + INT8 + benchmark), not off-the-shelf inference, to clear CompAg review.

## Result JSONs (Kaggle `/kaggle/working/`)
`phase0_result.json` (runs 2–4) · `phase0_probe_result.json` (run 5) · `phase0_specialize_result.json` (run 6) · `phase0_descriptors_result.json` (run 7, pending). Drop these next to `make_figures.py` to regenerate figures.
