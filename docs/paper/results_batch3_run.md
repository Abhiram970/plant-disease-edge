# EXP2 Results — Full Dataset (166 classes, 55,981 train / 13,938 test)
# From: python scripts/run_all_win.py --online --batch3 --batch32
# Date: July 14, 2026

## Probe Results (VLM zero-shot on seen classes)
| Model | Seen % | Linear-probe top1 |
|-------|--------|-------------------|
| MobileCLIP-S1 | 21.5% | 81.2% |
| MobileCLIP-S2 | 35.8% | 82.8% |
| MobileCLIP-B | 86.3% | 82.6% |

## EXP2: Train Seen / Keep Unseen (tier = MobileCLIP2-S0)
| Metric | Value |
|--------|-------|
| SEEN trained head | **82.6%** |
| SEEN zero-shot | 8.6% |
| UNSEEN zero-shot | **17.0%** |

### Unseen per-class breakdown:
| Crop | Accuracy |
|------|----------|
| Banana | 21% |
| Bean | 7% |
| Coffee | 21% |
| Cotton | 25% |
| Cucumber | 16% |
| Orange | 32% |

## EXP3: Fine-tune + WiSE-FT (tier = MobileCLIP2-S0, 55,981 seen images)
- epoch 1/5 loss=2.064 (in progress when screenshot taken)
