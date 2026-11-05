#!/usr/bin/env bash
# Canonical Skein training: structural fixed2, 5 seeds on 8x3090.
#
# Skein is the deployed model (formerly full_fixed2_aux / skein).  This
# wrapper keeps the historical training recipe: 800 epochs, no augmentation,
# fixed head 2, structural aux + Skeleton Recall + Tversky + high-res 512.
#
# Usage:
#   bash scripts/skein/run_skein_5seeds.sh
#
# Overrides:
#   EPOCHS=800 BASE_GPU=1 CKPT=checkpoints/sam_vit_l_0b3195.pth
#   CANONICAL_ROOT=results/runs LOG_DIR=results/ablation_logs
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

PY="${PYTHON:-/home/zkyd/anaconda3/envs/tsai/bin/python}"
EPOCHS="${EPOCHS:-800}"
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
  LOG="$LOG_DIR/skein_seed${SEED}_${STAMP}.log"
  echo "seed=$SEED gpu=$GPU log=$LOG"
  PYTHON="$PY" GPU="$GPU" SEED="$SEED" EPOCHS="$EPOCHS" \
  VAL_INTERVAL="$VAL_INTERVAL" CKPT="$CKPT" \
  TRAIN_ROOT="data/sacm_3shot_seed${SEED}" \
  CANONICAL_ROOT="$CANONICAL_ROOT" OUTPUT_ROOT= \
  NO_AUGMENT=1 USE_WANDB=true WANDB_MODE=online NUM_WORKERS=4 \
  WANDB_NAME="skein_seed${SEED}" \
    bash scripts/skein/run_variant.sh skein full \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done

RC=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || RC=1
done
exit "$RC"
