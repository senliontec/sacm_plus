#!/usr/bin/env bash
# full/fixed2 + structural supervision, 5 seeds on the 8x3090 server.
# Seeds 0-4 on GPUs 1-5 (GPU 0 left free), 300 epochs + 20-test eval.
#
# Usage: bash scripts/sacm_x/run_full_fixed2_aux_5seeds.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/zkyd/anaconda3/envs/tsai/bin/python}"
EPOCHS="${EPOCHS:-300}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
BASE_GPU="${BASE_GPU:-1}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
CANONICAL_ROOT="${CANONICAL_ROOT:-results/runs}"
LOG_DIR="${LOG_DIR:-results/ablation_logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
PIDS=()
for SEED in 0 1 2 3 4; do
  GPU=$((BASE_GPU + SEED))
  LOG="$LOG_DIR/fixed2_aux_seed${SEED}_${STAMP}.log"
  echo "seed=$SEED gpu=$GPU log=$LOG"
  PYTHON="$PY" GPU="$GPU" SEED="$SEED" EPOCHS="$EPOCHS" \
  VAL_INTERVAL="$VAL_INTERVAL" CKPT="$CKPT" \
  TRAIN_ROOT="data/sacm_3shot_seed${SEED}" \
  CANONICAL_ROOT="$CANONICAL_ROOT" OUTPUT_ROOT= \
  USE_WANDB=true WANDB_MODE=online NUM_WORKERS=4 \
  WANDB_NAME="sacm_x_fixed2_aux_seed${SEED}" \
    bash scripts/sacm_x/run_full_fixed2_aux.sh full \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done
RC=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || RC=1
done
exit "$RC"
