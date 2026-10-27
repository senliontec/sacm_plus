#!/usr/bin/env bash
# One-click Deep Closing shape-prior training on the protocol train masks.
#
# Usage:
#   GPU=0 bash scripts/sacm_x/train_deep_closing.sh
#   GPU=0 TRAIN_ROOT=data/sacm_3shot_seed1 \
#       OUTPUT=results/deep_closing/seed1.pth bash scripts/sacm_x/train_deep_closing.sh
#
# Environment:
#   GPU (default 0), TRAIN_ROOT (default data/sacm_3shot_seed0),
#   OUTPUT (default results/deep_closing/seed0_shape_prior.pth),
#   EPOCHS (300), RESOLUTION (512), BATCH_SIZE (1), LR (1e-2),
#   ALPHA (0.1), SEED (42), CHANNELS (16,32,64,128,256),
#   VAL_INTERVAL (10), LIMIT_TRAIN / LIMIT_VAL (optional smoke limits).
#
# This trains only on train/val masks; it never reads test masks.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

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
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot_seed0}"
OUTPUT="${OUTPUT:-results/deep_closing/seed0_shape_prior.pth}"
EPOCHS="${EPOCHS:-300}"
RESOLUTION="${RESOLUTION:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-2}"
ALPHA="${ALPHA:-0.1}"
SEED="${SEED:-42}"
CHANNELS="${CHANNELS:-16,32,64,128,256}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
LIMIT_TRAIN="${LIMIT_TRAIN:-}"
LIMIT_VAL="${LIMIT_VAL:-}"

if [ ! -f "$TRAIN_ROOT/protocol_manifest.csv" ]; then
  echo "ERROR: missing protocol manifest: $TRAIN_ROOT/protocol_manifest.csv" >&2
  exit 2
fi
if [ ! -d "$TRAIN_ROOT/train/masks" ]; then
  echo "ERROR: missing training masks: $TRAIN_ROOT/train/masks" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  DEVICE=cpu
else
  DEVICE=cuda
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

ARGS=(
  --data_root "$TRAIN_ROOT"
  --output "$OUTPUT"
  --epochs "$EPOCHS"
  --resolution "$RESOLUTION"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --alpha "$ALPHA"
  --seed "$SEED"
  --channels "$CHANNELS"
  --val_interval "$VAL_INTERVAL"
  --device "$DEVICE"
)
if [ -n "$LIMIT_TRAIN" ]; then
  ARGS+=(--limit_train "$LIMIT_TRAIN")
fi
if [ -n "$LIMIT_VAL" ]; then
  ARGS+=(--limit_val "$LIMIT_VAL")
fi

echo "Deep Closing shape prior: gpu=$GPU root=$TRAIN_ROOT output=$OUTPUT"
echo "epochs=$EPOCHS resolution=$RESOLUTION batch=$BATCH_SIZE lr=$LR alpha=$ALPHA"
PYTHONPATH=src "$PYTHON" scripts/sacm_x/train_deep_closing.py "${ARGS[@]}"
