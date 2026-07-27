# Plant-Disease-Edge — Final Results (13 July 2026)

> **Status: DONE** ✅ — Phases 1-8 complete. CNN baselines partial.
> Python: pyenv 3.11.9 | GPU: RTX 4060 Laptop 8GB | Data: 92,601 images, 20 crops, 217 classes

---

## COMPLETED ✅

### Phase 1 — Descriptors
- Heldout (Opus): 48/51 filled, 3 stubs
- All classes (Sonnet): 151/217 filled, 66 stubs
- 12 verified citations, 6 dead URLs

### Phase 2 — Descriptor Ablation (3 experiments)

| Exp | Classes | Crops | Chance |
|-----|---------|-------|--------|
| A | 16 | Coffee, Orange, Peach | 6.2% |
| B | 34 | + Cotton, Wheat, Bean | 2.9% |
| C | 51 | + Banana, Cucumber | 2.0% |

**Exp A (16 classes, chance=6.2%):**
| Model | bare | crude | rich | grounded |
|-------|------|-------|------|----------|
| S0 (11.4M) | 13.8% | 12.9% | 25.9% | 18.0% |
| S1 (21.5M) | 12.8% | 18.1% | 22.9% | 28.5% |
| S2 (35.8M) | 10.3% | 11.1% | **32.7%** | 13.7% |
| B (86.3M) | 14.5% | 16.8% | 30.7% | 26.4% |

**Exp B (34 classes, chance=2.9%):**
| Model | bare | crude | rich | grounded |
|-------|------|-------|------|----------|
| S0 | 20.7% | 17.1% | 20.9% | 20.8% |
| S1 | 18.4% | 19.6% | 23.8% | 26.1% |
| S2 | 17.8% | 15.1% | 23.1% | 19.3% |
| B | 24.4% | 23.9% | **29.4%** | 27.8% |

**Exp C (51 classes, chance=2.0%):**
| Model | bare | crude | rich | grounded |
|-------|------|-------|------|----------|
| S0 | 18.7% | 14.4% | 17.0% | **22.2%** |
| S1 | 16.5% | 17.1% | 20.1% | **24.5%** |
| S2 | 17.4% | 14.9% | 20.3% | **23.0%** |
| B | 23.3% | 20.5% | 24.2% | **29.1%** |

### Phase 3 — Top-5 + Abstain Metrics

**Exp A:**
| Model | Strategy | Top-1 | Top-5 | AURC |
|-------|----------|-------|-------|------|
| S0 | rich | 25.9% | 59.9% | 0.691 |
| S1 | grounded | 28.5% | 61.4% | **0.502** |
| S2 | rich | 32.7% | 64.5% | 0.608 |
| B | rich | 30.7% | 70.8% | 0.536 |
| SigLIP2 | rich | 31.6% | 69.1% | 0.659 |

**Exp B:**
| Model | Strategy | Top-1 | Top-5 | AURC |
|-------|----------|-------|-------|------|
| S1 | grounded | 26.1% | 66.3% | 0.563 |
| B | rich | 29.4% | 66.5% | **0.526** |
| SigLIP2 | rich | 28.6% | 73.0% | 0.591 |

**Exp C:**
| Model | Strategy | Top-1 | Top-5 | AURC |
|-------|----------|-------|-------|------|
| S1 | grounded | 24.5% | 63.0% | 0.620 |
| B | grounded | 29.1% | 76.0% | **0.588** |
| SigLIP2 | grounded | 28.9% | 73.3% | 0.624 |

### Phase 4 — LOCO (18 crops, 84,781 images)
| Crop | N | Acc | Role |
|------|---|-----|------|
| Orange | 1,361 | **29.8%** | held |
| Strawberry | 3,647 | 27.2% | train |
| Rose | 4,251 | 23.3% | train |
| Coffee | 1,582 | 14.4% | held |
| Banana | 2,068 | 12.9% | held |
| Grape | 5,342 | 10.9% | train |
| Sugarcane | 6,646 | 10.2% | train |
| Cucumber | 2,389 | 9.8% | held |
| **POOLED** | **84,781** | **7.7%** | — |

### Phase 5 — CNN Baselines (partial)

| Model | Params | Top-1 | Status |
|-------|--------|-------|--------|
| mobilenetv3_small_100 | 2.5M | **84.3%** | ✅ Saved |
| resnet50 | 25.6M | **88.4%** | ✅ Saved |

### Phase 6 — Edge Benchmark
| Model | Torch FP32 | ONNX FP32 | ONNX INT8 |
|-------|-----------|-----------|-----------|
| S0 (11.4M) | 80.7ms | 21.5ms | 463.2ms |
| S1 (21.5M) | 149.2ms | 40.1ms | 891.7ms |
| S2 (35.8M) | 178.0ms | 59.0ms | 1332.5ms |
| B (86.3M) | 174.8ms | 106.6ms | 70.7ms |

---

## TO COMPLETE ⏳ → MOVED TO DONE ✅

### Phase 7 — 6-Encoder Bake-off (NEW — 13 July 2026)

| Encoder | Params | Zero-shot | Coffee | Orange | Peach |
|---------|--------|-----------|--------|--------|-------|
| SigLIP2 | 92.9M | **30.4%** | 17% | 50% | 27% |
| MobileCLIP2-S2 | 35.8M | 30.1% | 30% | 45% | 12% |
| MobileCLIP2-S0 | 11.4M | 25.4% | 30% | 33% | 10% |
| MobileCLIP-S1 | 21.5M | 22.3% | 9% | 42% | 18% |
| SCOLD | 237.5M | 11.0% | 17% | 1% | 14% |
| BioCLIP2 | 304.0M | 6.3% | 6% | 8% | 4% |

### EXP2 (July 1) — Train Seen / Keep Unseen
- MobileCLIP2-S0 (11.4M), 80 seen classes, 17 unseen
- Seen probe: **67.2%** vs zero-shot 9.3%
- Unseen zero-shot: **27.0%** (Coffee 35%, Orange 29%, Peach 16%)

### EXP3 (July 1) — WiSE-FT Sweep
- α=0.0: Seen 67.0%, Unseen 27.0%
- α=0.5: Seen 77.6%, Unseen 15.4%
- α=1.0: Seen 82.2%, Unseen 10.4%

### Phase 8 — Figures (13 July 2026)
11 figures generated in `docs/paper/figures/`:
bakeoff, descriptor_ablation, scaling, edge_pareto, efficiency_curve, hybrid, riskcoverage, wiseft, specialization_forgetting, arch_comparison, descriptors

### Remaining (optional — not blocking paper)
- [ ] 4 CNN baselines (efficientnetv2, mobilenetv4, convnextv2, fastvit) — blocked by Windows workers issue
- [ ] VLM probe_seen.py — unblocked, can run anytime

## RESULT FILES ON DISK
All in `C:\kaggle\working\results\`:
- zeroshot_eval_A.json, zeroshot_eval_B.json, zeroshot_eval_C.json
- metrics_abstain_A.json, metrics_abstain_B.json, metrics_abstain_C.json
- loco_s0_rich.json
- supervised_mobilenetv3_small_100.json, supervised_resnet50.json
- edge_benchmark.json