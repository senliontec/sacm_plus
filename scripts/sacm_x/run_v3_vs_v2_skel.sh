#!/usr/bin/env bash
# One-click decision run: full_v2_skel vs full_v3.
#
# Default: 300 epochs, train + validation only (no test-set evaluation),
# same seed, two GPUs in parallel.  Choose the winner on validation macro
# Dice / clDice, then run the 800-epoch x 5-seed full benchmark.
#
# Usage:
#   bash scripts/sacm_x/run_v3_vs_v2_skel.sh
#   GPU_V2=1 GPU_V3=2 EPOCHS=300 SEED=0 bash scripts/sacm_x/run_v3_vs_v2_skel.sh
#   MODE=full bash scripts/sacm_x/run_v3_vs_v2_skel.sh   # 800 epochs + 20-test eval
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# Default placement: GPU 0 is reserved for other work.
GPU_V2="${GPU_V2:-1}"
GPU_V3="${GPU_V3:-2}"
MODE="${MODE:-train}"          # train | full
if [ "$MODE" = "full" ]; then
  EPOCHS="${EPOCHS:-800}"
else
  EPOCHS="${EPOCHS:-300}"
fi
SEED="${SEED:-0}"
VAL_INTERVAL="${VAL_INTERVAL:-5}"
MODE="${MODE:-train}"          # train | full
CANONICAL_ROOT="${CANONICAL_ROOT:-results/runs}"
LOG_DIR="${LOG_DIR:-results/ablation_logs}"
USE_WANDB="${USE_WANDB:-true}"

case "$MODE" in
  train) SKIP_TEST=true ;;
  full)  EPOCHS="${EPOCHS:-800}"; SKIP_TEST=false ;;
  *) echo "MODE must be train | full" >&2; exit 2 ;;
esac

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_V2="$LOG_DIR/v2_skel_seed${SEED}_${STAMP}.log"
LOG_V3="$LOG_DIR/v3_seed${SEED}_${STAMP}.log"

echo "============================================================"
echo " v2_skel vs v3"
echo " gpus:      $GPU_V2 (v2_skel) / $GPU_V3 (v3)"
echo " epochs:    $EPOCHS"
echo " seed:      $SEED"
echo " mode:      $MODE (skip_test=$SKIP_TEST)"
echo " val every: $VAL_INTERVAL"
echo " logs:      $LOG_V2"
echo "            $LOG_V3"
echo "============================================================"

GPU="$GPU_V2" SEED="$SEED" EPOCHS="$EPOCHS" VAL_INTERVAL="$VAL_INTERVAL" \
CANONICAL_ROOT="$CANONICAL_ROOT" USE_WANDB="$USE_WANDB" \
  bash scripts/sacm_x/run_pepa_variant.sh full_v2_skel "$MODE" \
  > "$LOG_V2" 2>&1 &
PID_V2=$!

GPU="$GPU_V3" SEED="$SEED" EPOCHS="$EPOCHS" VAL_INTERVAL="$VAL_INTERVAL" \
CANONICAL_ROOT="$CANONICAL_ROOT" USE_WANDB="$USE_WANDB" \
  bash scripts/sacm_x/run_full_v3.sh "$MODE" \
  > "$LOG_V3" 2>&1 &
PID_V3=$!

set +e
wait "$PID_V2"; RC_V2=$?
wait "$PID_V3"; RC_V3=$?
set -e

echo
echo "v2_skel exit=$RC_V2  log=$LOG_V2"
echo "v3      exit=$RC_V3  log=$LOG_V3"
if [ "$RC_V2" -ne 0 ] || [ "$RC_V3" -ne 0 ]; then
  echo "At least one run failed; inspect the logs above." >&2
  exit 1
fi

echo
echo "Compare validation metrics:"
echo "  results/runs/sacm_x/full_v2_skel/seed_${SEED}/metrics_history.json"
echo "  results/runs/sacm_x/full_v3/seed_${SEED}/metrics_history.json"
