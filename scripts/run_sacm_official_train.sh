#!/usr/bin/env bash
# One-click official SACM baseline training through the unified runner.
#
# Usage:
#   bash scripts/run_sacm_official_train.sh
#   GPU=3 EPOCHS=60 bash scripts/run_sacm_official_train.sh
#
# Environment:
#   PYTHON, GPU, EPOCHS, BATCH_SIZE, ADAPTER_RATIO, ADAPTER_LR,
#   WEIGHT_DECAY, BCE_WEIGHT, DICE_WEIGHT, DATA_ROOT, SAM_CKPT, WORKDIR,
#   SEED, USE_WANDB, WANDB_MODE
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
source scripts/wandb_env.sh
GPU="${GPU:-0}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ADAPTER_RATIO="${ADAPTER_RATIO:-0.1}"
ADAPTER_LR="${ADAPTER_LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BCE_WEIGHT="${BCE_WEIGHT:-0.7}"
DICE_WEIGHT="${DICE_WEIGHT:-0.3}"
DATA_ROOT="${DATA_ROOT:-$REPO/data/sacm_3shot}"
SAM_CKPT="${SAM_CKPT:-$REPO/checkpoints/sam_vit_l_0b3195.pth}"
WORKDIR="${WORKDIR:-$REPO/workdirs/baselines/sacm_official}"
SEED="${SEED:-42}"

if [ ! -f "$SAM_CKPT" ]; then
  echo "ERROR: missing SAM ViT-L checkpoint: $SAM_CKPT" >&2
  exit 2
fi

CMD=("$PYTHON" -m baseline_zoo.run_one --model sacm_official --stage train
     --data-root "$DATA_ROOT" --workdir "$WORKDIR"
     --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
     --lr "$ADAPTER_LR" --weight-decay "$WEIGHT_DECAY"
     --adapter-dim-ratio "$ADAPTER_RATIO"
     --bce-weight "$BCE_WEIGHT" --dice-weight "$DICE_WEIGHT"
     --grad-clip 1.0 --seed "$SEED" --sam-checkpoint "$SAM_CKPT"
     --device cuda)
if [ "$USE_WANDB" = "true" ]; then
  CMD+=(--use_wandb --wandb_project "$WANDB_PROJECT"
        --wandb_entity "$WANDB_ENTITY" --wandb_host "$WANDB_BASE_URL"
        --wandb_mode "$WANDB_MODE")
fi

echo "gpu=$GPU data_root=$DATA_ROOT epochs=$EPOCHS workdir=$WORKDIR"
CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
