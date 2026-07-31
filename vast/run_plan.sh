#!/usr/bin/env bash
# ==============================================================================
# Vast.ai master run plan — spend a small GPU budget in strict value order.
#
# Designed for a ~$4.60 budget on a 24GB GPU (RTX 3090 / A5000 ~ $0.20-0.25/hr).
# EVERY stage is RESUMABLE and SKIPS work whose result JSON already exists, so if
# the box dies or credit runs out you lose nothing but the current stage.
#
#   bash vast/run_plan.sh            # run all stages in priority order
#   bash vast/run_plan.sh 2 3        # run only stages 2 and 3
#   STAGES="1 2" bash vast/run_plan.sh
#
# Stages are ordered so the PAPER-BLOCKING work lands first. If you run out of
# money after stage 3 you still have everything you need to finish the draft.
# ==============================================================================
set -u

cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"

# ---- environment -------------------------------------------------------------
export PDE_DATA_ROOT="${PDE_DATA_ROOT:-/workspace/working}"
export PDE_DATASET_DIR="${PDE_DATASET_DIR:-/workspace/working/exp_data}"
RESULTS="$PDE_DATA_ROOT/results"
LOGS="$REPO/logs/vast"
mkdir -p "$RESULTS" "$LOGS" "$PDE_DATASET_DIR"

if [ -f /tmp/PDEPY ]; then PY="$(cat /tmp/PDEPY)"; else PY="python"; fi
echo "[plan] PY=$PY"
echo "[plan] DATA=$PDE_DATASET_DIR  RESULTS=$RESULTS"

START_TS=$(date +%s)
banner() {
  local now; now=$(date +%s)
  echo ""
  echo "=============================================================="
  echo "  STAGE $1 — $2"
  echo "  elapsed: $(( (now-START_TS)/60 )) min"
  echo "=============================================================="
}
# run a stage, tee to a log, never abort the whole plan on failure
run() {  # run <logname> <cmd...>
  local name="$1"; shift
  echo "[run] $* "
  if "$@" 2>&1 | tee "$LOGS/$name.log"; then
    echo "[ok] $name"
  else
    echo "[FAIL] $name — continuing with the next stage (see $LOGS/$name.log)"
  fi
}
have() { [ -s "$RESULTS/$1" ]; }   # result JSON already exists and is non-empty

STAGES="${STAGES:-${*:-1 2 3 4 5 6}}"
echo "[plan] stages to run: $STAGES"

# ==============================================================================
# STAGE 1 — data + manifest        (~30-45 min, ~$0.15)   MANDATORY
# ==============================================================================
if [[ " $STAGES " == *" 1 "* ]]; then
  banner 1 "SAGE data fetch + manifest"
  # sage_data.py is incremental: it keeps .shards_done.json and only pulls missing shards.
  # 13 parquet shards (~10GB each); it auto-stops once every wanted crop is covered.
  run "01_sage_data"     "$PY" scripts/sage_data.py --role all
  run "01_manifest"      "$PY" scripts/build_manifest.py --min-images 25
  echo "[plan] image count: $(find "$PDE_DATASET_DIR" -type f -name '*.jpg' 2>/dev/null | wc -l)"
fi

# ==============================================================================
# STAGE 2 — EXP3 WiSE-FT full sweep   (~1-1.5 h, ~$0.35)   *** P0: BLOCKS PAPER ***
# The alpha table currently in the paper is from the OLD small config and the run
# on full data stopped at epoch 1/5. This finishes it.
# ==============================================================================
if [[ " $STAGES " == *" 2 "* ]]; then
  banner 2 "EXP3 fine-tune + WiSE-FT alpha sweep (P0 — blocks the paper)"
  run "02_exp3" "$PY" kaggle/run_all_win.py --only exp3 --tier lw11 --ft-epochs 5 --batch 64
fi

# ==============================================================================
# STAGE 3 — scale study A/B/C        (~45-60 min, ~$0.20)  *** P0: fixes the
# 27%-vs-17% inconsistency by producing ALL configs from ONE consistent sweep ***
# ==============================================================================
if [[ " $STAGES " == *" 3 "* ]]; then
  banner 3 "Zero-shot scale study across Experiments A / B / C (P0 — canonical table)"
  for EXP in A B C; do
    run "03_zeroshot_exp$EXP" "$PY" scripts/evaluate.py \
        --exp "$EXP" --strategies bare crude rich grounded \
        --tiers lw11 lw21 lw35 --heavy --teachers
    # keep each experiment's JSON separately so they can't overwrite each other
    [ -f "$RESULTS/zeroshot_eval.json" ] && cp "$RESULTS/zeroshot_eval.json" \
        "$RESULTS/zeroshot_eval_exp$EXP.json"
  done
fi

# ==============================================================================
# STAGE 4 — seen-head probe across A/B/C   (~45-60 min, ~$0.25)
# Gives the "seen accuracy vs #classes" half of the scale study.
# ==============================================================================
if [[ " $STAGES " == *" 4 "* ]]; then
  banner 4 "Seen-crop linear probe across A / B / C"
  for EXP in A B C; do
    run "04_probe_exp$EXP" "$PY" scripts/probe_seen.py --exp "$EXP" --epochs 40
    [ -f "$RESULTS/probe_seen.json" ] && cp "$RESULTS/probe_seen.json" \
        "$RESULTS/probe_seen_exp$EXP.json"
  done
fi

# ==============================================================================
# STAGE 5 — supervised CNN baselines   (~2-2.5 h, ~$0.55)
# Only mobilenetv3_small_100 is done. This is the "CNNs are strong on seen but
# STRUCTURALLY 0 on unseen" table. Deliberately a SUBSET - 6 well-chosen archs
# beat 13 half-finished ones, and the runner skips anything already done.
# ==============================================================================
if [[ " $STAGES " == *" 5 "* ]]; then
  banner 5 "Supervised CNN baseline family (resumable; skips completed archs)"
  run "05_cnn_baselines" "$PY" scripts/run_cnn_baselines.py \
      --archs resnet50 mobilenetv3_large_100 mobilenetv4_conv_small \
              efficientnet_b0 convnextv2_nano fastvit_t8 \
      --epochs 12 --batch 128 --workers 8
fi

# ==============================================================================
# STAGE 6 — LOCO + abstain metrics    (~30-40 min, ~$0.15)  rebuttal-proofing
# ==============================================================================
if [[ " $STAGES " == *" 6 "* ]]; then
  banner 6 "LOCO (anti-cherry-pick) + top-5/abstain metrics"
  run "06_loco" "$PY" scripts/loco.py --model s0 --strategy rich --bootstrap 2000
  run "06_metrics" "$PY" scripts/metrics.py --models s0 s1 s2 b \
      --strategies rich grounded --reference
fi

# ==============================================================================
# WRAP UP — collect everything into one folder to download
# ==============================================================================
banner "*" "COLLECTING RESULTS"
OUT="$REPO/vast_results"
mkdir -p "$OUT"
# scripts/ write to $RESULTS ; kaggle/run_all_win.py writes to ./exp_out (no /kaggle/working here)
cp -v "$RESULTS"/*.json    "$OUT/" 2>/dev/null
cp -v "$REPO"/exp_out/*.json "$OUT/" 2>/dev/null
cp -rv "$LOGS" "$OUT/logs" 2>/dev/null
tar -czf "$REPO/vast_results.tar.gz" -C "$REPO" vast_results 2>/dev/null

echo "[collect] JSONs gathered: $(ls -1 "$OUT"/*.json 2>/dev/null | wc -l)"
if [ ! -s "$REPO/vast_results.tar.gz" ]; then
  echo "[collect] !! tarball empty — check the stage logs before destroying the instance"
fi

echo ""
echo "=============================================================="
echo "  DONE — total elapsed $(( ($(date +%s)-START_TS)/60 )) min"
echo "  Results : $OUT"
echo "  Tarball : $REPO/vast_results.tar.gz   <-- download this"
echo "=============================================================="
ls -la "$OUT" 2>/dev/null | head -40
