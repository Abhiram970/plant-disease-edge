#!/usr/bin/env python
"""
Edge quantization benchmark + REGRESSION DIAGNOSIS  (Kaggle-ready, CPU only, no GPU needed)

WHY THIS EXISTS
---------------
`scripts/benchmark_edge.py` reported INT8 as 21-23x SLOWER than FP32 for the lightweight tier:

    model            onnx fp32     onnx int8      result
    MobileCLIP2-S0    21.5 ms      463.2 ms      21.6x SLOWER
    MobileCLIP-S1     40.1 ms      891.7 ms      22.2x SLOWER
    MobileCLIP2-S2    59.0 ms     1332.5 ms      22.6x SLOWER
    MobileCLIP-B     106.6 ms       70.7 ms       1.5x faster  <-- only the pure ViT

Model SIZE compressed correctly (~3.8x), so quantization did apply -- the problem is the
RUNTIME. Root-cause hypothesis: the old code called `quantize_dynamic(...)` with defaults.
Dynamic quantization is designed for weight-dominated MatMul stacks (plain transformers).
MobileCLIP S0/S1/S2 are FastViT-style HYBRIDS with heavy depthwise convolutions; ORT cannot
fuse those into QLinearConv, so it emits per-inference DynamicQuantizeLinear + ConvInteger /
falls back to de-quantize->float->re-quantize for most of the graph. MobileCLIP-B is a pure
ViT (all MatMul) -> dynamic quant works, hence the 1.5x speedup.

WHAT THIS SCRIPT DOES
---------------------
For each model it builds and times FIVE variants and prints an op-level diagnosis:
  1. torch FP32          (baseline)
  2. ONNX FP32           (the honest deployment number today)
  3. ONNX FP16           (usually the best size/speed tradeoff on CPU+NPU)
  4. ONNX INT8 dynamic   (the OLD broken path -- kept to reproduce the regression)
  5. ONNX INT8 static    (QDQ + per-channel + calibration + quant_pre_process -- the FIX)
plus a node histogram (QLinearConv vs ConvInteger vs DynamicQuantizeLinear) so we can PROVE
which path fused and which fell back.

HOW TO RUN ON KAGGLE
--------------------
  New Notebook -> Accelerator: None (CPU) -> Internet: ON
  Upload this file, then in a cell:      %run benchmark_quantization.py
  Quick smoke run:                       %run benchmark_quantization.py --models s0 --runs 20
Results are written to /kaggle/working/edge_quant_benchmark.{json,md}

IMPORTANT: latency MUST be measured on the deployment target. Kaggle CPU is a valid
"laptop-class" row. Re-run the same file on a Raspberry Pi for the Pi row.
Calibration here uses random tensors -- fine for LATENCY. For an accuracy-vs-INT8 claim,
re-run static quantization with real images (--calib-dir).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")           # fp16 truncation spam
logging.getLogger().setLevel(logging.ERROR)  # per-weight quantization notes

# --------------------------------------------------------------------------- deps
def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


try:
    import onnx  # noqa
    import onnxruntime as ort  # noqa
except ImportError:
    print("[setup] installing onnx / onnxruntime ...")
    pip("onnx", "onnxruntime")
try:
    import open_clip  # noqa
except ImportError:
    print("[setup] installing open_clip_torch ...")
    pip("open_clip_torch")

import numpy as np
import onnx
import onnxruntime as ort
import torch

# --------------------------------------------------------------------------- config
# (mirrors scripts/config.py DEPLOY_MODELS so numbers are comparable to the repo)
MODELS = {
    "s0": ("MobileCLIP2-S0", "dfndr2b"),     # ~11.4M -> small NPU / phone
    "s1": ("MobileCLIP-S1",  "datacompdr"),  # ~21.5M -> laptop CPU
    "s2": ("MobileCLIP2-S2", "dfndr2b"),     # ~35.8M -> laptop
    "b":  ("MobileCLIP-B",   "datacompdr"),  # ~86.3M -> workstation
}
IMG_SIZE = 224
_KAGGLE = Path("/kaggle/working")
# on Windows "/kaggle/working" resolves against the current drive, so require posix
OUT_DIR = _KAGGLE if (os.name == "posix" and _KAGGLE.exists()) else Path("./results")
ONNX_DIR = OUT_DIR / "onnx"

# Integer compute kernels present in the serialized graph.
# NOTE: ConvInteger/MatMulInteger are NOT necessarily fast -- they still need re-scaling.
INT8_OPS = {"QLinearConv", "QLinearMatMul", "MatMulInteger", "ConvInteger", "QGemm"}
# Quantize/dequantize conversion nodes. In QDQ format these are EXPECTED: ORT fuses them
# into QLinear* kernels at session-init, which a static graph scan cannot see. So node
# counts alone CANNOT decide "fused vs fallback" -- only measured latency can.
CONVERT_OPS = {"DynamicQuantizeLinear", "QuantizeLinear", "DequantizeLinear"}


# --------------------------------------------------------------------------- helpers
def percentiles(ts):
    ts = sorted(ts)
    def pct(p):
        return ts[min(len(ts) - 1, int(round(p / 100 * (len(ts) - 1))))]
    return {"p50_ms": round(pct(50) * 1e3, 2),
            "p95_ms": round(pct(95) * 1e3, 2),
            "mean_ms": round(sum(ts) / len(ts) * 1e3, 2)}


def session(path, threads=None):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def time_onnx(path, runs, warmup=5, threads=None):
    sess = session(path, threads)
    x = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    name = sess.get_inputs()[0].name
    # fp16 graphs want fp16 input
    if sess.get_inputs()[0].type == "tensor(float16)":
        x = x.astype(np.float16)
    xin = {name: x}
    for _ in range(warmup):
        sess.run(None, xin)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, xin)
        ts.append(time.perf_counter() - t0)
    return {**percentiles(ts), "size_mb": round(Path(path).stat().st_size / 1e6, 2)}


def op_histogram(path):
    m = onnx.load(str(path))
    return Counter(n.op_type for n in m.graph.node)


def diagnose(hist, p50_ms=None, fp32_p50_ms=None):
    """Node counts are EVIDENCE; the verdict comes from measured latency vs ONNX FP32,
    because ORT fuses QDQ pairs at session-init (invisible in the serialized graph)."""
    d = {"int8_kernels": sum(v for k, v in hist.items() if k in INT8_OPS),
         "convert_nodes": sum(v for k, v in hist.items() if k in CONVERT_OPS),
         "float_conv_left": hist.get("Conv", 0),
         "float_matmul_left": hist.get("MatMul", 0)}
    if p50_ms and fp32_p50_ms:
        sp = fp32_p50_ms / p50_ms                      # >1 = faster than fp32
        d["speedup_vs_fp32"] = round(sp, 2)
        d["verdict"] = ("FASTER than fp32" if sp >= 1.1 else
                        "parity with fp32" if sp >= 0.9 else
                        f"SLOWER than fp32 ({1/sp:.1f}x) - not viable")
    else:
        d["verdict"] = "n/a"
    return d


# --------------------------------------------------------------------------- builders
def export_fp32(visual, path):
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    visual.eval()
    with torch.no_grad():
        torch.onnx.export(visual, x, str(path), input_names=["image"], output_names=["emb"],
                          opset_version=17, dynamic_axes={"image": {0: "batch"}})
    return path


def preprocess(fp32_path, pre_path):
    """Shape inference + graph cleanup. Skipping this is a main reason quantization
    fails to fuse -- the old benchmark never called it."""
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=False)
        return pre_path
    except Exception as e:
        print(f"      [warn] quant_pre_process failed ({type(e).__name__}); using raw fp32 graph")
        return fp32_path


def build_fp16(fp32_path, out):
    try:
        from onnxconverter_common import float16
    except ImportError:
        pip("onnxconverter-common")
        try:
            from onnxconverter_common import float16
        except Exception:
            return None
    try:
        m16 = float16.convert_float_to_float16(onnx.load(str(fp32_path)), keep_io_types=False)
        onnx.save(m16, str(out))
        return out
    except Exception as e:
        print(f"      [warn] fp16 convert failed: {type(e).__name__}: {str(e)[:70]}")
        return None


def build_int8_dynamic(pre_path, out):
    """The OLD path (reproduces the regression)."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(pre_path), str(out), weight_type=QuantType.QInt8)
    return out


def build_int8_static(pre_path, out, calib_n, calib_dir=None):
    """The FIX: QDQ format + per-channel weights + real calibration."""
    from onnxruntime.quantization import (quantize_static, QuantType, QuantFormat,
                                          CalibrationDataReader)

    sess = session(pre_path)
    in_name = sess.get_inputs()[0].name

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.data = self._load()
            self.i = 0

        def _load(self):
            if calib_dir and Path(calib_dir).exists():
                from PIL import Image
                paths = [p for p in Path(calib_dir).rglob("*")
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")][:calib_n]
                out = []
                for p in paths:
                    im = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    a = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
                    out.append(a)
                if out:
                    print(f"      calibrating on {len(out)} real images")
                    return out
            print(f"      calibrating on {calib_n} random tensors (latency-valid; "
                  f"use --calib-dir for accuracy claims)")
            return [np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
                    for _ in range(calib_n)]

        def get_next(self):
            if self.i >= len(self.data):
                return None
            d = {in_name: self.data[self.i]}
            self.i += 1
            return d

    quantize_static(
        str(pre_path), str(out), Reader(),
        quant_format=QuantFormat.QDQ,      # QDQ lets ORT fuse into QLinear* kernels
        per_channel=True,                  # per-channel weights: needed for depthwise conv
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    return out


# --------------------------------------------------------------------------- per-model
def run_model(key, runs, calib_n, calib_dir, threads):
    name, pretrained = MODELS[key]
    print(f"\n{'='*70}\n[{key}] {name} ({pretrained})\n{'='*70}")
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"model": name, "pretrained": pretrained, "variants": {}, "diagnosis": {}}

    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    visual = model.visual.eval()
    entry["img_params_M"] = round(sum(p.numel() for p in visual.parameters()) / 1e6, 2)
    print(f"  image-encoder params: {entry['img_params_M']}M")

    # 1) torch fp32
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        for _ in range(5):
            visual(x)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter(); visual(x); ts.append(time.perf_counter() - t0)
    entry["variants"]["torch_fp32"] = percentiles(ts)
    print(f"  torch  fp32        {entry['variants']['torch_fp32']['p50_ms']:>9.2f} ms")

    # 2) onnx fp32 (+ preprocessed graph for the quantizers)
    fp32 = export_fp32(visual, ONNX_DIR / f"{key}_fp32.onnx")
    pre = preprocess(fp32, ONNX_DIR / f"{key}_pre.onnx")
    entry["variants"]["onnx_fp32"] = time_onnx(fp32, runs, threads=threads)
    print(f"  onnx   fp32        {entry['variants']['onnx_fp32']['p50_ms']:>9.2f} ms "
          f"({entry['variants']['onnx_fp32']['size_mb']} MB)")

    # 3) fp16
    fp16 = build_fp16(fp32, ONNX_DIR / f"{key}_fp16.onnx")
    if fp16:
        try:
            entry["variants"]["onnx_fp16"] = time_onnx(fp16, runs, threads=threads)
            print(f"  onnx   fp16        {entry['variants']['onnx_fp16']['p50_ms']:>9.2f} ms "
                  f"({entry['variants']['onnx_fp16']['size_mb']} MB)")
        except Exception as e:
            print(f"  onnx   fp16        FAILED ({type(e).__name__})")

    fp32_p50 = entry["variants"]["onnx_fp32"]["p50_ms"]

    # 4) int8 dynamic (old, broken path)
    try:
        d8 = build_int8_dynamic(pre, ONNX_DIR / f"{key}_int8_dynamic.onnx")
        entry["variants"]["onnx_int8_dynamic"] = time_onnx(d8, runs, threads=threads)
        entry["diagnosis"]["int8_dynamic"] = diagnose(
            op_histogram(d8), entry["variants"]["onnx_int8_dynamic"]["p50_ms"], fp32_p50)
        v = entry["variants"]["onnx_int8_dynamic"]
        print(f"  onnx   int8-dyn    {v['p50_ms']:>9.2f} ms ({v['size_mb']} MB)  "
              f"-> {entry['diagnosis']['int8_dynamic']['verdict']}")
    except Exception as e:
        print(f"  onnx   int8-dyn    FAILED ({type(e).__name__}: {str(e)[:60]})")

    # 5) int8 static QDQ (the fix)
    try:
        s8 = build_int8_static(pre, ONNX_DIR / f"{key}_int8_static.onnx", calib_n, calib_dir)
        entry["variants"]["onnx_int8_static"] = time_onnx(s8, runs, threads=threads)
        entry["diagnosis"]["int8_static"] = diagnose(
            op_histogram(s8), entry["variants"]["onnx_int8_static"]["p50_ms"], fp32_p50)
        v = entry["variants"]["onnx_int8_static"]
        print(f"  onnx   int8-static {v['p50_ms']:>9.2f} ms ({v['size_mb']} MB)  "
              f"-> {entry['diagnosis']['int8_static']['verdict']}")
    except Exception as e:
        print(f"  onnx   int8-static FAILED ({type(e).__name__}: {str(e)[:60]})")

    # speedups vs onnx fp32
    base = entry["variants"]["onnx_fp32"]["p50_ms"]
    entry["speedup_vs_onnx_fp32"] = {
        k: round(base / v["p50_ms"], 2)
        for k, v in entry["variants"].items() if k != "torch_fp32" and "p50_ms" in v
    }
    del model, visual
    return entry


# --------------------------------------------------------------------------- report
def write_report(table, out_md):
    lines = ["# Edge quantization benchmark", "",
             f"- device: CPU · img {IMG_SIZE}x{IMG_SIZE} · batch 1 · runs {table['runs']}",
             f"- threads: {table['threads']}", "",
             "## Latency p50 (ms) — lower is better", "",
             "| model | params | torch fp32 | onnx fp32 | onnx fp16 | int8 dynamic | int8 static |",
             "|---|---|---|---|---|---|---|"]
    for key, e in table["models"].items():
        if "variants" not in e:
            continue
        g = lambda k: (f"{e['variants'][k]['p50_ms']:.1f}" if k in e["variants"] else "—")
        lines.append(f"| {e['model']} | {e.get('img_params_M','?')}M | {g('torch_fp32')} | "
                     f"{g('onnx_fp32')} | {g('onnx_fp16')} | {g('onnx_int8_dynamic')} | "
                     f"{g('onnx_int8_static')} |")
    lines += ["", "## Size (MB)", "",
              "| model | onnx fp32 | onnx fp16 | int8 dynamic | int8 static |", "|---|---|---|---|---|"]
    for key, e in table["models"].items():
        if "variants" not in e:
            continue
        s = lambda k: (f"{e['variants'][k].get('size_mb','—')}" if k in e["variants"] else "—")
        lines.append(f"| {e['model']} | {s('onnx_fp32')} | {s('onnx_fp16')} | "
                     f"{s('onnx_int8_dynamic')} | {s('onnx_int8_static')} |")
    lines += ["", "## Quantization diagnosis", "",
              "Verdict is decided by **measured latency vs ONNX FP32** — node counts are evidence only,",
              "because ORT fuses QDQ pairs at session-init (not visible in the serialized graph).", "",
              "| model | path | int8 kernels | convert nodes | float conv left | speedup vs fp32 | verdict |",
              "|---|---|---|---|---|---|---|"]
    for key, e in table["models"].items():
        for path, d in e.get("diagnosis", {}).items():
            lines.append(f"| {e['model']} | {path} | {d['int8_kernels']} | {d['convert_nodes']} "
                         f"| {d['float_conv_left']} | {d.get('speedup_vs_fp32','—')} | {d['verdict']} |")
    lines += ["", "## How to read this", "",
              "- **int8-static faster than fp32** → the old regression was a tooling bug; report INT8.",
              "- **int8-static still slower than fp32** → INT8 is not viable for these hybrid",
              "  conv-transformer backbones on the ORT CPU EP. **Report ONNX FP32 as the deployment",
              "  number** and present INT8 purely as a *size* tradeoff (~3.5x smaller, slower).",
              "  That is an honest, publishable finding, not a failure.", ""]
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Edge quantization benchmark + diagnosis")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--calib-n", type=int, default=32, help="calibration samples for static quant")
    ap.add_argument("--calib-dir", default=None, help="dir of real images for calibration (accuracy)")
    ap.add_argument("--threads", type=int, default=None, help="intra_op threads (default: ORT auto)")
    # tolerate Jupyter's injected -f argument
    args, _ = ap.parse_known_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"ORT {ort.__version__} | torch {torch.__version__} | out={OUT_DIR}")
    table = {"device": "cpu", "img_size": IMG_SIZE, "runs": args.runs,
             "threads": args.threads or "auto", "ort_version": ort.__version__, "models": {}}

    for key in args.models:
        try:
            table["models"][key] = run_model(key, args.runs, args.calib_n, args.calib_dir, args.threads)
        except Exception as e:
            print(f"[{key}] FAILED: {type(e).__name__}: {str(e)[:120]}")
            table["models"][key] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}

    (OUT_DIR / "edge_quant_benchmark.json").write_text(json.dumps(table, indent=2))
    write_report(table, OUT_DIR / "edge_quant_benchmark.md")
    print(f"\nsaved:\n  {OUT_DIR/'edge_quant_benchmark.json'}\n  {OUT_DIR/'edge_quant_benchmark.md'}")
    print("\nNext: copy both files into docs/paper/ and commit.")


if __name__ == "__main__":
    main()
