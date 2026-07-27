# Kaggle runners

Self-contained scripts you **upload to a Kaggle notebook and `%run`** — no repo clone, no auth.

| File | What it runs | Accelerator | Internet |
|---|---|---|---|
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
