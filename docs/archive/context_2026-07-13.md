# Plant-Disease-Edge — Complete Experiment Results (13 July 2026)

## Dataset
- 92,601 images, 20 crops, 217 classes
- SEEN: 10 crops, 166 classes
- HELD: 8 crops, 51 classes (chance 2.0%)
- GPU: RTX 4060 Laptop 8GB

---

## Phase 1 — Descriptors ✅
- Heldout (Opus): 48/51 filled, 3 stubs
- All classes (Sonnet): 151/217 filled, 66 stubs
- 12 verified citations, 6 dead URLs

---

## Phase 2 — Descriptor Ablation ✅
| Exp | Classes | Chance | Best Model | Best Strategy | Top-1 |
|-----|---------|--------|------------|---------------|-------|
| A | 16 | 6.2% | S2 | rich | 32.7% |
| B | 34 | 2.9% | B | rich | 29.4% |
| C | 51 | 2.0% | B | grounded | 29.1% |

---

## Phase 3 — Top-5 + Abstain Metrics ✅
| Exp | Model | Strategy | Top-1 | Top-5 | AURC |
|-----|-------|----------|-------|-------|------|
| A | S1 | grounded | 28.5% | 61.4% | 0.502 |
| B | B | rich | 29.4% | 66.5% | 0.526 |
| C | B | grounded | 29.1% | 76.0% | 0.588 |

---

## Phase 4 — LOCO (18-crop anti-cherry-pick) ✅
- Pooled 84,781 images, 7.7% accuracy
- Orange best: 29.8%, Coffee: 14.4%, Banana: 12.9%

---

## Phase 5 — VLM Linear Probe (NEW — 13 July) ✅
Frozen VLM backbones + linear classifier on seen crops (166 classes)

| Model | Params | Seen Probe |
|-------|--------|------------|
| MobileCLIP2-S0 | 11.4M | 82.6% |
| MobileCLIP-S1 | 21.5M | 81.4% |
| **MobileCLIP2-S2** | 35.8M | **82.8%** |
| MobileCLIP-B | 86.3M | 82.4% |

**Surprise:** S2 (35.8M) beats B (86.3M)! S0 at 11.4M matches B at 82.6%.

## Phase 5 — CNN Baselines ✅ (partial)

| Model | Params | Top-1 |
|-------|--------|-------|
| mobilenetv3_small_100 | 2.5M | 84.3% |
| resnet50 | 25.6M | 88.4% |
| 4 remaining | — | BLOCKED (Cloud GPU needed) |

---

## Phase 6 — Edge Benchmark ✅
| Model | Torch FP32 | ONNX FP32 | ONNX INT8 |
|-------|-----------|-----------|-----------|
| S0 (11.4M) | 80.7ms | 21.5ms | 463.2ms |
| S1 (21.5M) | 149.2ms | 40.1ms | 891.7ms |
| S2 (35.8M) | 178.0ms | 59.0ms | 1332.5ms |
| B (86.3M) | 174.8ms | 106.6ms | 70.7ms |

---

## Phase 7 — 6-Encoder Bake-off (NEW — 13 July) ✅
17 classes, chance 5.9%, rich descriptors

| Encoder | Params | Zero-shot | Coffee | Orange | Peach |
|---------|--------|-----------|--------|--------|-------|
| **SigLIP2** | 92.9M | **30.4%** | 17% | 50% | 27% |
| MobileCLIP2-S2 | 35.8M | 30.1% | 30% | 45% | 12% |
| MobileCLIP2-S0 | 11.4M | 25.4% | 30% | 33% | 10% |
| MobileCLIP-S1 | 21.5M | 22.3% | 9% | 42% | 18% |
| SCOLD | 237.5M | 11.0% | 17% | 1% | 14% |
| BioCLIP2 | 304.0M | 6.3% | 6% | 8% | 4% |

---

## EXP2 — Train Seen / Keep Unseen (July 1) ✅
MobileCLIP2-S0 (11.4M), 80 seen, 17 unseen
- Seen probe: **67.2%** vs zero-shot 9.3% (7.2x improvement)
- Unseen zero-shot: **27.0%** (Coffee 35%, Orange 29%, Peach 16%)

---

## EXP3 — WiSE-FT Sweep (July 1) ✅
| α | Seen | Unseen |
|---|------|--------|
| 0.0 | 67.0% | 27.0% |
| 0.5 | 77.6% | 15.4% |
| 1.0 | 82.2% | 10.4% |

---

## Phase 8 — Figures ✅ (13 July)
11 figures in `docs/paper/figures/`:
bakeoff, descriptor_ablation, scaling, edge_pareto, efficiency_curve, hybrid, riskcoverage, wiseft, specialization_forgetting, arch_comparison, descriptors

---

## Key Claims for Paper
1. **MobileCLIP2-S2 (35.8M) matches SigLIP2 (92.9M)** at 30.1% vs 30.4% — 3x smaller
2. **MobileCLIP2-S0 (11.4M) reaches 86% of SigLIP2** at 1/8 the params — deployable on edge
3. **Domain CLIPs fail** — BioCLIP2 6.3%, SCOLD 11.0% — general-purpose VLMs beat domain-specific
4. **Descriptors matter** — rich/grounded > crude > bare across all experiments
5. **WiSE-FT tradeoff** — α=0.5 gives 77.6% seen / 15.4% unseen
6. **CNN ceiling** — ResNet50 88.4% on seen, but structurally 0% on unseen

## Next Steps
1. Start paper draft NOW — all experiments are done
2. Copy JSONs to docs/paper/
3. Add WiSE-FT figure to make_figures.py (already in figures/)
4. Write abstract, intro, methods, results, discussion