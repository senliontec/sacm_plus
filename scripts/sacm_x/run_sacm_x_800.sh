#!/usr/bin/env bash
# One-click 800-epoch SACM-X training with W&B on the server.
#
# Default: ViT-L, preset=full, seed=0, GPU 0, 800 epochs, validation every
# epoch (VAL_INTERVAL=1, same cadence as the baseline runtime), W&B online.
# Test evaluation is intentionally skipped: this script is for the long
# training curve only.
#
# Usage:
#   bash scripts/sacm_x/run_sacm_x_800.sh
#   GPU=1 SEED=1 bash scripts/sacm_x/run_sacm_x_800.sh
#   PRESET=full_srec GPU=2 bash scripts/sacm_x/run_sacm_x_800.sh
#   PRESET=stage2 GPU=0 bash scripts/sacm_x/run_sacm_x_800.sh   # 24GB fallback
#   MODEL=sam_b CKPT=checkpoints/sam_vit_b_01ec64.pth \
#     TRAIN_ROOT=data/sacm_3shot bash scripts/sacm_x/run_sacm_x_800.sh
#
# Optional environment:
#   PYTHON, GPU=0, MODEL=sam_l, PRESET=full, EPOCHS=800,
#   VAL_INTERVAL=1, SEED=0,
#   CKPT=checkpoints/sam_vit_l_0b3195.pth,
#   TRAIN_ROOT=data/sacm_3shot_seed${SEED},
#   OUTPUT_ROOT=results/sacm_x_vitl_800/seed${SEED},
#   MONITOR=official, USE_WANDB=true, WANDB_MODE=online,
#   WANDB_GROUP=sacm_x_${MODEL}_${PRESET}_800_seed${SEED}
#
# Outputs:
#   <OUTPUT_ROOT>/<PRESET>/best_model.pth
#   <OUTPUT_ROOT>/<PRESET>/metrics_history.json
#   W&B run under entity buaazqk / project sacm
#
# Notes:
#   - `full` is the main SACM-X model.  If a 24GB server card reports OOM,
#     use `PRESET=stage2` (geometry adapters + coarse-to-fine, no
#     fusion-v2/multi-depth) or `MODEL=sam_b`.
#   - `PRESET=full_srec` adds the Skeleton Recall topology loss.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi

GPU="${GPU:-0}"
MODEL="${MODEL:-sam_l}"
PRESET="${PRESET:-full}"
EPOCHS="${EPOCHS:-800}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
SEED="${SEED:-0}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/sacm_x_vitl_800/seed${SEED}}"
MONITOR="${MONITOR:-official}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-sacm_x_${MODEL}_${PRESET}_800_seed${SEED}}"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: missing SAM checkpoint: $CKPT" >&2
  exit 2
fi
if [ ! -f "$TRAIN_ROOT/protocol_manifest.csv" ]; then
  echo "ERROR: missing protocol: $TRAIN_ROOT/protocol_manifest.csv" >&2
  echo "Set TRAIN_ROOT to the seed-specific 3-shot split root." >&2
  exit 2
fi
if [ "$USE_WANDB" = "true" ] && [ "$WANDB_MODE" != "offline" ]; then
  # shellcheck disable=SC1091
  source scripts/sacm_x/wandb_env.sh
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is online but no credentials were found." >&2
    echo "Set WANDB_API_KEY or ~/.netrc, or use WANDB_MODE=offline." >&2
    exit 2
  fi
  export WANDB_RUN_GROUP="$WANDB_GROUP"
fi

echo "SACM-X 800-epoch training"
echo "  model=$MODEL preset=$PRESET epochs=$EPOCHS val_interval=$VAL_INTERVAL"
echo "  seed=$SEED gpu=$GPU monitor=$MONITOR test=skipped"
echo "  checkpoint=$CKPT"
echo "  train_root=$TRAIN_ROOT"
echo "  output=$OUTPUT_ROOT/$PRESET"
if [ "$USE_WANDB" = "true" ]; then
  echo "  wandb=online group=$WANDB_GROUP"
else
  echo "  wandb=disabled"
fi

GPU="$GPU" \
MODEL="$MODEL" \
PRESET="$PRESET" \
EPOCHS="$EPOCHS" \
VAL_INTERVAL="$VAL_INTERVAL" \
SEED="$SEED" \
CKPT="$CKPT" \
TRAIN_ROOT="$TRAIN_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
MONITOR="$MONITOR" \
USE_WANDB="$USE_WANDB" \
WANDB_MODE="$WANDB_MODE" \
WANDB_GROUP="$WANDB_GROUP" \
SKIP_TEST=true \
SKIP_AGGREGATE=true \
bash scripts/sacm_x/run_sacm_x_quick.sh
