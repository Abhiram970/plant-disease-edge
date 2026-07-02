## Plant-Disease-Edge — Status as of 1 July 2026

### Architecture
Frozen compact CLIP family: MobileCLIP2-S0 (11M), S1 (21M), S2 (35M) + SigLIP2 (86M), BioCLIP2 (93M). Trained seen head + descriptor zero-shot unseen + WiSE-FT + abstain.

### Bake-off Results (Jun 29)
SigLIP2 31.5% (best) · S2 28.7% · S0 26.9% · S1 22.5% · BioCLIP2 9.6% (dropped)

### Hybrid Validated
One 11M backbone → 67% seen (trained) + 27% unseen (zero-shot), simultaneously.

### Bugs Found & Fixed
1. EXP3 theta0 deepcopy bug (alpha=0 was forgotten model) — fixed
2. SCOLD adapter mis-driving (5.3% = fake score, RoBERTa fell back to base weights) — low priority since SigLIP2 confirmed best

### ALL THREE EXPERIMENTS — COMPLETE with full Coffee data (Jul 1) ✅
Local RTX 4060. Held set = Coffee+Orange+Peach (17 classes, chance 5.9%), rich descriptors.
Data: SAGE shards [0,1,2,3,4,5,8]; Coffee=579, Orange=662, Peach=588, Soybean=5590,
Apple=2663, Potato=1368, Corn=837. Seen fine-tune on 8,054 imgs × 5 epochs
(loss 3.523→2.238→1.515→1.072→0.785).

**EXP1 — encoder bake-off** (run_all_bakeoff.json)
| Encoder | Params | Zero-shot | Coffee | Orange | Peach |
|---|---|---|---|---|---|
| SigLIP2 | 92.9M | 31.5% | 21% | 37% | 37% |
| MobileCLIP2-S2 | 35.8M | 28.7% | 30% | 36% | 20% |
| MobileCLIP2-S0 | 11.4M | 27.0% | 35% | 29% | 16% |
| MobileCLIP-S1 | 21.5M | 22.4% | 14% | 29% | 24% |
| BioCLIP2 | 304.0M | 9.6% | 15% | 4% | 9% |
| SCOLD | 237.5M | 4.4% | 3% | 5% | 5% (below chance = broken wrapper) |

**EXP2 — train seen / keep unseen** (run_all_train_seen_lw11.json), MobileCLIP2-S0, 80 seen / 17 unseen
- SEEN trained head 67.2% vs zero-shot 9.3% (7.2× lift)
- UNSEEN zero-shot 27.0% preserved (Coffee 35%, Orange 29%, Peach 16%)

**EXP3 — WiSE-FT sweep** (run_all_exp3_lw11.json), MobileCLIP2-S0
| Alpha | SEEN | UNSEEN |
|---|---|---|
| 0.0 (frozen)        | 67.0% | 27.0% |
| 0.5 (WiSE-FT)       | 77.6% | 15.4% |
| 1.0 (full finetune) | 82.2% | 10.4% |

Consistency: alpha=0 (67.0/27.0) ≡ EXP2 trained head (67.2) and ≡ EXP1 S0 unseen (27.0,
same 35/29/16 per-crop) → three-way anchor, theta0 fix confirmed. Monotonic tradeoff
(seen↑ / unseen↓). Non-monotonic vs size: S0 27.0 > S1 22.4 (MobileCLIP2 dfndr2b beats
older MobileCLIP datacompdr). BioCLIP2/SCOLD (domain CLIPs) fail — general CLIPs win.
Note: alpha=0.5/1.0 wobble ±1-2pp run-to-run (prior run: 77.3/16.8, 80.6/9.3); alpha=0 stable.

Artifacts at C:\kaggle\working\ (ROOT resolved there): run_all_bakeoff.json,
run_all_train_seen_lw11.json, run_all_exp3_lw11.json, run_all_ft_lw11.pt, run_all_probe_lw11.pt.
Repo docs/paper/ copies are STALE (Jun-29 8-class) — copy the new ones in and commit.

### Environment (Jul 1)
- torch 2.5.1+cu121 on pyenv Python 3.11.9, RTX 4060 8GB (cuda=True)
- `temp/run_all_win.py` = Windows-safe copy (num_workers 2→0); OOM fix batch_size 512→64, cols=None fallback removed
- Full SAGE data now on disk locally (all 3 held crops covered incl. Coffee)

### Pending / Next
- Re-run EXP1 bake-off + EXP2 on the SAME Coffee-inclusive snapshot (committed
  `run_all_bakeoff.json` is a stale 8-class Orange+Peach run) — cheap, data already local
- Copy/commit `run_all_exp3_lw11.json` into the repo; update `docs/paper/make_figures.py`
  with the WiSE-FT tradeoff curve + numbers
- Optional: descriptor ablation (rich vs crude vs bare) — core "source-grounded" claim
- Optional: per-tier sweep (lw21, lw35) if claiming a per-tier deployable table
- Open question answered: full finetune (80.6%) DOES beat the 67% linear probe on seen,
  but at the cost of unseen (9.3%); WiSE-FT alpha=0.5 keeps 16.8% unseen at 77.3% seen
