# Edge quantization benchmark

- device: CPU · img 224x224 · batch 1 · runs 50
- threads: auto

## Latency p50 (ms) — lower is better

| model | params | torch fp32 | onnx fp32 | onnx fp16 | int8 dynamic | int8 static |
|---|---|---|---|---|---|---|
| MobileCLIP2-S0 | 11.41M | 47.1 | 17.4 | — | 320.0 | 62.2 |
| MobileCLIP-S1 | 21.54M | 99.8 | 33.7 | — | 614.9 | 79.5 |
| MobileCLIP2-S2 | 35.82M | 119.0 | 49.2 | — | 923.8 | 98.7 |
| MobileCLIP-B | 86.35M | 130.7 | 100.8 | — | 60.4 | 46.4 |

## Size (MB)

| model | onnx fp32 | onnx fp16 | int8 dynamic | int8 static |
|---|---|---|---|---|
| MobileCLIP2-S0 | 45.81 | — | 12.1 | 12.92 |
| MobileCLIP-S1 | 86.51 | — | 22.81 | 24.29 |
| MobileCLIP2-S2 | 143.62 | — | 37.31 | 39.16 |
| MobileCLIP-B | 345.55 | — | 87.42 | 87.4 |

## Quantization diagnosis

Verdict is decided by **measured latency vs ONNX FP32** — node counts are evidence only,
because ORT fuses QDQ pairs at session-init (not visible in the serialized graph).

| model | path | int8 kernels | convert nodes | float conv left | speedup vs fp32 | verdict |
|---|---|---|---|---|---|---|
| MobileCLIP2-S0 | int8_dynamic | 124 | 100 | 0 | 0.05 | SLOWER than fp32 (18.4x) - not viable |
| MobileCLIP2-S0 | int8_static | 0 | 1290 | 119 | 0.28 | SLOWER than fp32 (3.6x) - not viable |
| MobileCLIP-S1 | int8_dynamic | 224 | 182 | 0 | 0.05 | SLOWER than fp32 (18.2x) - not viable |
| MobileCLIP-S1 | int8_static | 0 | 2340 | 215 | 0.42 | SLOWER than fp32 (2.4x) - not viable |
| MobileCLIP2-S2 | int8_dynamic | 244 | 198 | 0 | 0.05 | SLOWER than fp32 (18.8x) - not viable |
| MobileCLIP2-S2 | int8_static | 0 | 2536 | 235 | 0.5 | SLOWER than fp32 (2.0x) - not viable |
| MobileCLIP-B | int8_dynamic | 52 | 52 | 0 | 1.67 | FASTER than fp32 |
| MobileCLIP-B | int8_static | 0 | 1013 | 3 | 2.17 | FASTER than fp32 |

## How to read this

- **int8-static faster than fp32** → the old regression was a tooling bug; report INT8.
- **int8-static still slower than fp32** → INT8 is not viable for these hybrid
  conv-transformer backbones on the ORT CPU EP. **Report ONNX FP32 as the deployment
  number** and present INT8 purely as a *size* tradeoff (~3.5x smaller, slower).
  That is an honest, publishable finding, not a failure.
