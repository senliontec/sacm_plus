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
VAL_INTERVAL="${VAL_INTERVAL:-1}"
# Dedicated root: never inherit CANONICAL_ROOT from an older audit shell.
ABLATION_ROOT="${ABLATION_ROOT:-results/runs/ablation_v3_vs_v2_skel}"
# Base SAM weights by default.  Do not inherit CKPT from an audit shell:
# a results/runs/.../best_model.pth is a training artifact, not a raw SAM
# state dict, and would silently warm-start from the wrong checkpoint.
SAM_CKPT="${ABLATION_CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
TRAIN_ROOT="${ABLATION_TRAIN_ROOT:-data/sacm_3shot_seed${SEED}}"
LOG_DIR="${LOG_DIR:-results/ablation_logs}"
# W&B is online by default.  Do not inherit USE_WANDB=false from an older
# smoke/audit shell; disable explicitly with ABLATION_USE_WANDB=false.
USE_WANDB="${ABLATION_USE_WANDB:-true}"
WANDB_MODE="${ABLATION_WANDB_MODE:-online}"

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
echo " wandb:     $USE_WANDB (mode=$WANDB_MODE)"
echo " checkpoint: $SAM_CKPT"
echo " train root: $TRAIN_ROOT"
echo " out root:  $ABLATION_ROOT"
echo " logs:      $LOG_V2"
echo "            $LOG_V3"
echo "============================================================"

GPU="$GPU_V2" SEED="$SEED" EPOCHS="$EPOCHS" VAL_INTERVAL="$VAL_INTERVAL" \
CKPT="$SAM_CKPT" TRAIN_ROOT="$TRAIN_ROOT" \
CANONICAL_ROOT="$ABLATION_ROOT" OUTPUT_ROOT= WANDB_NAME= \
USE_WANDB="$USE_WANDB" WANDB_MODE="$WANDB_MODE" \
  bash scripts/sacm_x/run_pepa_variant.sh full_v2_skel "$MODE" \
  > "$LOG_V2" 2>&1 &
PID_V2=$!

GPU="$GPU_V3" SEED="$SEED" EPOCHS="$EPOCHS" VAL_INTERVAL="$VAL_INTERVAL" \
CKPT="$SAM_CKPT" TRAIN_ROOT="$TRAIN_ROOT" \
CANONICAL_ROOT="$ABLATION_ROOT" OUTPUT_ROOT= WANDB_NAME= \
USE_WANDB="$USE_WANDB" WANDB_MODE="$WANDB_MODE" \
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
  echo
  echo "===== v2_skel log tail ====="
  tail -n 80 "$LOG_V2" || true
  echo
  echo "===== v3 log tail ====="
  tail -n 80 "$LOG_V3" || true
  echo
  echo "At least one run failed; full logs:"
  echo "  $LOG_V2"
  echo "  $LOG_V3"
  exit 1
fi

V2_HIST="${ABLATION_ROOT}/sacm_x/full_v2_skel/seed_${SEED}/metrics_history.json"
V3_HIST="${ABLATION_ROOT}/sacm_x/full_v3/seed_${SEED}/metrics_history.json"
PYTHON="${PYTHON:-python}"
if [ -f "$V2_HIST" ] && [ -f "$V3_HIST" ]; then
  "$PYTHON" scripts/sacm_x/compare_val_runs.py \
    "full_v2_skel=$V2_HIST" "full_v3=$V3_HIST" || true
fi

echo
echo "Validation histories:"
echo "  $V2_HIST"
echo "  $V3_HIST"
