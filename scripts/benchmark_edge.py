"""
Phase D (minimum viable) — on-device efficiency table for the deployable image encoders.

Only the IMAGE encoder ships. For each of the 4 models this measures, on THIS machine's CPU
(a valid "laptop" tier number; run the same script on a Raspberry Pi for the Pi row):
  - params (M) and, if `thop`/`ptflops` present, MACs (GFLOPs)
  - FP32 latency p50/p95 (torch, CPU, batch 1, 224x224)
  - ONNX FP32 latency (onnxruntime) + on-disk size
  - INT8 latency (onnxruntime dynamic quantization) + on-disk size

Produces the accuracy-vs-params-vs-latency Pareto inputs the CompAg reviewer expects for an
"at the edge" claim. GPU is NOT used — latency must be measured on the deployment target.

DEPS (CPU only):  pip install onnx onnxruntime
                  pip install thop         # optional, for MACs

RASPBERRY PI:  copy the exported int8 .onnx files + this repo, then on the Pi run
               `python scripts/benchmark_edge.py --models s0 s1 --onnx-only` (uses the .onnx
               already in RESULTS_DIR/onnx, skips torch export) to get the Pi latency row.

USAGE
  python scripts/benchmark_edge.py --models s0 s1 s2 b
  python scripts/benchmark_edge.py --models s0 --runs 100
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def _percentiles(ts):
    ts = sorted(ts)
    def pct(p):
        return ts[min(len(ts) - 1, int(round(p / 100 * (len(ts) - 1))))]
    return {"p50_ms": round(pct(50) * 1e3, 2), "p95_ms": round(pct(95) * 1e3, 2),
            "mean_ms": round(sum(ts) / len(ts) * 1e3, 2)}


def torch_latency(visual, runs, warmup=5):
    import torch
    visual.eval()
    x = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    with torch.no_grad():
        for _ in range(warmup):
            visual(x)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter(); visual(x); ts.append(time.perf_counter() - t0)
    return _percentiles(ts)


def macs_g(visual):
    try:
        import torch
        from thop import profile
        x = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
        macs, _ = profile(visual, inputs=(x,), verbose=False)
        return round(macs / 1e9, 3)
    except Exception:
        return None


def export_and_time_onnx(visual, name, runs, onnx_dir):
    """Export visual->onnx, quantize to int8, return {fp32:{...}, int8:{...}} latency + sizes."""
    import torch
    onnx_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = onnx_dir / f"{name}_visual_fp32.onnx"
    int8_path = onnx_dir / f"{name}_visual_int8.onnx"
    x = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    visual.eval()
    torch.onnx.export(visual, x, str(fp32_path), input_names=["image"], output_names=["emb"],
                      opset_version=17, dynamic_axes={"image": {0: "batch"}})
    result = {}
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        for tag, path in (("fp32", fp32_path), ("int8", int8_path)):
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            xin = {sess.get_inputs()[0].name: x.numpy()}
            for _ in range(5):
                sess.run(None, xin)
            ts = []
            for _ in range(runs):
                t0 = time.perf_counter(); sess.run(None, xin); ts.append(time.perf_counter() - t0)
            result[tag] = {**_percentiles(ts), "size_mb": round(path.stat().st_size / 1e6, 2)}
    except Exception as e:
        result["onnx_error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(C.DEPLOY_MODELS))
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--no-onnx", action="store_true", help="skip ONNX export/INT8 (torch latency only)")
    args = ap.parse_args()

    import open_clip
    import torch
    torch.set_num_threads(torch.get_num_threads())  # respect machine default
    onnx_dir = C.RESULTS_DIR / "onnx"
    models = C.resolve_models(args.models)
    table = {"device": "cpu", "img_size": C.IMG_SIZE, "runs": args.runs, "models": {}}

    for name, pretrained in models:
        print(f"\n[edge] {name} ({pretrained}) ...")
        try:
            model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
            visual = model.visual.eval()
            params_m = round(sum(p.numel() for p in visual.parameters()) / 1e6, 2)
            entry = {"img_params_M": params_m, "macs_G": macs_g(visual),
                     "torch_fp32": torch_latency(visual, args.runs)}
            print(f"   params={params_m}M  torch_fp32 p50={entry['torch_fp32']['p50_ms']}ms "
                  f"p95={entry['torch_fp32']['p95_ms']}ms")
            if not args.no_onnx:
                entry["onnx"] = export_and_time_onnx(visual, name, args.runs, onnx_dir)
                o = entry["onnx"]
                if "int8" in o:
                    print(f"   onnx fp32 {o['fp32']['p50_ms']}ms/{o['fp32']['size_mb']}MB  "
                          f"int8 {o['int8']['p50_ms']}ms/{o['int8']['size_mb']}MB")
            table["models"][name] = entry
            del model, visual
        except Exception as e:
            print(f"   skipped ({type(e).__name__}: {str(e)[:80]})")
            table["models"][name] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS_DIR / "edge_benchmark.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\n[edge] saved {out}")
    print("[edge] run this same script on a Raspberry Pi for the Pi latency row.")


if __name__ == "__main__":
    main()
