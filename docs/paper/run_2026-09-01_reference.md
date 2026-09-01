# Reference numbers from the 2026-09-01 session (NOT paper data)

**These are scraped from a Kaggle log, not read from a result JSON. Nothing here may be cited.**
Their only purpose is a drift check: when the re-run finishes, its generated JSONs should reproduce
these values. A disagreement is itself worth investigating — silently differing numbers would mean
something non-deterministic in the pipeline that we have not accounted for.

Session: commit `ae43cec`, 84,123 images / 18 crops / 217 classes, SAGE pinned `bc9bd2899f19`,
descriptors from `claude-sonnet-5` via Lava. Reached LOCO at t+3.8 h; the CNN sweep was cut short.

## Known-invalid in this session

- **Control arm (all three seeds).** `descriptors.UNGROUNDED_SEED` was read at import while
  `evaluate.py` sets `PDE_UNGROUNDED_SEED` afterwards, so every seed evaluated
  `descriptors_ungrounded/0`. The three seeds returned byte-identical numbers. Fixed in `a518889`.
- **`grounded_matched`.** 43 of 51 classes filled per seed; the other 8 fell through to `rich`.
  The JSON repair in `cabfc9a` addresses the cause; `MIN_FILLED` is now 50.

Both are excluded below.

## Zero-shot, mean over the four deployable encoders

| scale | classes | chance | bare | crude | rich | grounded |
|---|---|---|---|---|---|---|
| A | 16 | 6.2 % | 12.7 % | 14.1 % | 28.9 % | 19.4 % |
| B | 34 | 2.9 % | 20.4 % | 18.7 % | 25.0 % | 22.1 % |
| C | 51 | 2.0 % | 19.1 % | 16.7 % | 21.5 % | 23.5 % |

Means are over the four DEPLOYABLE encoders; ViT-B-16-SigLIP2 is the reference ceiling and is
excluded, as in the manuscript. (Computed twice: a first pass mixed the control-arm rows into scale
C because the block bounds were wrong, giving 23.7/21.7/27.5/30.2. The bounds are now anchored on
each scale's own `[eval] saved` line.)

Per encoder at scale C: S0 `bare 20.2 / crude 15.6 / rich 19.3 / grounded 21.0`;
S1 `16.2 / 16.4 / 20.7 / 23.5`; S2 `18.1 / 15.4 / 21.5 / 22.7`; B `22.0 / 19.3 / 24.6 / 26.9`.

## Abstention and top-5 (5 models × 3 scales)

Scale C, 14,204 images, 51 classes, chance 2.0 %:

| model | rich top-1 / top-5 / AURC | grounded top-1 / top-5 / AURC |
|---|---|---|
| MobileCLIP2-S0 11.4 M | 19.3 / 61.7 / 0.775 | 21.0 / 61.8 / 0.728 |
| MobileCLIP-S1 21.5 M | 20.7 / 58.1 / 0.701 | 23.5 / 59.1 / 0.672 |
| MobileCLIP2-S2 35.8 M | 21.5 / 66.1 / 0.697 | 22.7 / 67.8 / 0.740 |
| MobileCLIP-B 86.3 M | 24.6 / 66.0 / 0.655 | 26.9 / 74.3 / 0.620 |
| ViT-B-16-SigLIP2 92.9 M | 23.1 / 67.1 / 0.681 | 28.0 / 68.8 / 0.627 |

Scale A (16 classes, chance 6.2 %) and B (34 classes, chance 2.9 %) are in the session log.

## Seen-crop linear probe

| scale | classes | images | S0 11.4 M | S1 21.5 M | S2 35.8 M | B 86.3 M |
|---|---|---|---|---|---|---|
| A | 97 | 42,326 | 78.7 % | 78.4 % | 79.3 % | 79.2 % |
| B | 154 | 62,043 | 80.7 % | 79.3 % | 80.6 % | 80.6 % |
| C | 166 | 69,919 | 82.2 % | 81.1 % | 82.2 % | 82.1 % |

Accuracy RISES as the label space grows (97 → 166 classes) and spans ~1.1 pp across a 7.6× parameter
range — the flatness result, reproduced.

## Leave-one-crop-out

MobileCLIP2-S0, `rich`, 26,151 images, 78 classes, chance 1.3 %, 2,000 bootstrap samples:

| crop | n | acc | 95 % CI | role |
|---|---|---|---|---|
| Orange | 1,361 | 28.9 % | 26.5–31.2 | held |
| Coffee | 1,582 | 15.8 % | 14.0–17.6 | held |
| Apple | 10,000 | 12.7 % | 12.0–13.3 | train-pool |
| Peach | 1,107 | 12.3 % | 10.4–14.3 | held |
| Potato | 2,698 | 9.5 % | 8.3–10.6 | train-pool |
| Corn | 9,403 | 4.8 % | 4.4–5.3 | train-pool |

Held-out crops are not systematically easier than train-pool crops — two of the three held crops sit
mid-range and one train-pool crop is the weakest — which is the anti-cherry-picking check.

## Supervised CNN

`tf_efficientnetv2_s` completed at batch 32 after two OOM retries. `convnextv2_tiny` reached epoch 3
at 85.5 % before the fragmentation OOM fixed in `72dced4`.
