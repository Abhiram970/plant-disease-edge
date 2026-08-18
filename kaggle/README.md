# Kaggle runners

## Start here: `run_everything.py`

Paste the whole file as **one cell**. It reproduces the entire study — SAGE fetch, cross-crop
zero-shot at three scales, abstention, seen-crop probe, leave-one-crop-out, the label-corrected
sensitivity run, and all 14 supervised CNN baselines.

    Accelerator = GPU T4 x2 (or P100) · Internet = ON · Persistence = Files only
    Add-ons -> Secrets -> GH_TOKEN  (GitHub fine-grained PAT, read-only Contents)
    Save Version -> "Save & Run All (Commit)", then close the tab.

Roughly 9-11 h for everything; the stage flags at the top let you run a subset. Every stage skips
work whose result file already exists, the embedding cache resumes mid-encoder, and CNNs resume from
their last epoch checkpoint — so if the session wall stops it, just run it again.

It verifies the fetch before training on it: if the pull comes back short of ~60k images over 16
crops it stops with an explanation rather than spending eight hours producing class counts that will
not match the paper.

Attach a previously published `pde-sage-data` dataset (or any attached folder containing
`exp_data/`) to skip the 40-90 min fetch. `manifest.csv` is always rebuilt, never reused, because it
stores absolute image paths and an attached dataset mounts somewhere else.

---

## Older, single-purpose runners

| File | What it runs | Accelerator | Internet |
|---|---|---|---|
| `cnn_baselines_finish.py` | Just the CNN architectures a previous sweep missed | GPU | ON |
| `run_all.py` | EXP1 encoder bake-off · EXP2 hybrid (train seen / keep unseen) · EXP3 fine-tune + WiSE-FT | **GPU** | ON |
| `run_all_win.py` | Same, tuned for the Windows/local box | GPU | ON |
| `bakeoff.py` | EXP1 only (encoder bake-off) | GPU | ON |
| `train_seen.py` | EXP2 only (seen-crop head) | GPU | ON |
| `benchmark_quantization.py` | **Edge latency + INT8 regression diagnosis** | **None (CPU)** | ON |

---

## benchmark_quantization.py — run this next (P0-1)

**Why:** the committed `docs/paper/edge_benchmark.json` reports INT8 as **21–23× slower** than FP32
for S0/S1/S2, while only the pure-ViT MobileCLIP-B got faster (1.5×). Sizes compressed correctly, so
quantization applied — the runtime is falling back. The old code called `quantize_dynamic()` with
defaults: no `quant_pre_process`, no `per_channel`, no calibration. Dynamic quant suits
weight-dominated MatMul stacks (plain ViT); MobileCLIP S0/S1/S2 are FastViT-style **hybrids with
depthwise convs**, which ORT can't fuse that way.

**Run it:**
```
New Notebook → Accelerator: None (CPU) → Internet: ON
Upload benchmark_quantization.py
%run benchmark_quantization.py                       # all 4 models, ~10-20 min
%run benchmark_quantization.py --models s0 --runs 20 # quick smoke first
```

It builds and times **five** variants per model — torch FP32, ONNX FP32, ONNX FP16, INT8-dynamic
(the old broken path, kept to reproduce the bug), and **INT8-static QDQ + per-channel + calibration**
(the fix) — then prints a node histogram verdict per graph:

- `fused-int8` → real `QLinearConv`/`QLinearMatMul` kernels. The fix worked.
- `FALLBACK (convert-per-inference)` → `DynamicQuantizeLinear`/`Dequantize` everywhere. Still broken.

**Outputs:** `/kaggle/working/edge_quant_benchmark.json` + `.md`. Download both, drop into
`docs/paper/`, and commit.

### How to decide from the result

| Outcome | What we report in the paper |
|---|---|
| int8-static is fast + `fused-int8` | Use INT8 as the deployment number. Regression was a tooling bug. |
| both int8 paths still `FALLBACK` | **Report ONNX FP32/FP16** and document INT8 as a limitation. This is still a strong claim — S0 at ~21.5 ms CPU ≈ 46 FPS on an 11M model. |

Either way we get a defensible efficiency table. What we must **not** do is publish the current
"INT8 is 22× slower" number.

### Notes
- Latency must be measured on the deployment target. Kaggle CPU = a valid *laptop-class* row.
  Re-run the same file on a **Raspberry Pi** for the Pi row.
- Calibration defaults to random tensors — fine for **latency**. For an INT8 **accuracy** claim,
  pass `--calib-dir /path/to/images` to calibrate on real crop images.
- Only the **image encoder** is benchmarked; the text encoder runs offline to precompute prototypes.
