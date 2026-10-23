#!/usr/bin/env bash
# One-click training for the official SACM baseline (SAM ViT-L + adapters).
#
# The official script writes best_model.pth in its current directory, so this
# wrapper runs it inside the workdir and then adds split provenance metadata.
#
# Usage:
#   bash scripts/run_sacm_official_train.sh                 # 60 epochs
#   EPOCHS=5 VAL_INTERVAL=1 bash scripts/run_sacm_official_train.sh
#   GPU=3 bash scripts/run_sacm_official_train.sh
#
# Environment:
#   PYTHON, GPU, EPOCHS, VAL_INTERVAL, BATCH_SIZE (keep 1), ADAPTER_RATIO,
#   ADAPTER_LR, DECODER_LR, WEIGHT_DECAY, BCE_WEIGHT, DICE_WEIGHT,
#   DATA_ROOT, SAM_CKPT, WORKDIR, LOG_DIR
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-60}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ADAPTER_RATIO="${ADAPTER_RATIO:-0.1}"
ADAPTER_LR="${ADAPTER_LR:-3e-4}"
DECODER_LR="${DECODER_LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BCE_WEIGHT="${BCE_WEIGHT:-0.7}"
DICE_WEIGHT="${DICE_WEIGHT:-0.3}"
DATA_ROOT="${DATA_ROOT:-$REPO/data/sacm_3shot}"
SAM_CKPT="${SAM_CKPT:-$REPO/checkpoints/sam_vit_l_0b3195.pth}"
WORKDIR="${WORKDIR:-$REPO/workdirs/baselines/sacm_official}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"

if [ ! -f "$SAM_CKPT" ]; then
  echo "ERROR: missing SAM ViT-L checkpoint: $SAM_CKPT" >&2
  exit 2
fi
if [ ! -d "$DATA_ROOT/train/images" ] || [ ! -d "$DATA_ROOT/train/masks" ]; then
  echo "ERROR: expected $DATA_ROOT/train/{images,masks}" >&2
  exit 2
fi
if [ "$BATCH_SIZE" != "1" ]; then
  echo "WARNING: official zero-prompt tensors are allocated with batch_size;" >&2
  echo "         batch_size=1 is strongly recommended for 18 images." >&2
fi

mkdir -p "$WORKDIR" "$LOG_DIR"
LOG="$LOG_DIR/sacm_official_train_$(date +%Y%m%d_%H%M%S).log"
export PYTHONPATH="$REPO/baselines/sacm${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "data_root=$DATA_ROOT"
echo "sam_checkpoint=$SAM_CKPT"
echo "workdir=$WORKDIR"
echo "epochs=$EPOCHS val_interval=$VAL_INTERVAL batch_size=$BATCH_SIZE"
echo "log=$LOG"

cd "$WORKDIR"
"$PYTHON" "$REPO/baselines/sacm/train_sam.py" \
  --data_root "$DATA_ROOT" \
  --checkpoint "$SAM_CKPT" \
  --batch_size "$BATCH_SIZE" \
  --adapter_dim_ratio "$ADAPTER_RATIO" \
  --adapter_lr "$ADAPTER_LR" \
  --decoder_lr "$DECODER_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --epochs "$EPOCHS" \
  --val_interval "$VAL_INTERVAL" \
  --bce_weight "$BCE_WEIGHT" \
  --dice_weight "$DICE_WEIGHT" \
  --scheduler cosine 2>&1 | tee "$LOG"

if [ ! -f "$WORKDIR/best_model.pth" ]; then
  echo "ERROR: training finished but $WORKDIR/best_model.pth is missing." >&2
  echo "       With EPOCHS=$EPOCHS and VAL_INTERVAL=$VAL_INTERVAL no validation" >&2
  echo "       may have run; choose VAL_INTERVAL <= EPOCHS." >&2
  exit 1
fi

# Add split provenance so run_baseline_test.py can validate the checkpoint.
"$PYTHON" - "$WORKDIR/best_model.pth" "$DATA_ROOT/split_manifest.csv" "$ADAPTER_RATIO" <<'PY'
import hashlib
import sys
import torch
from pathlib import Path

ckpt_path = Path(sys.argv[1])
split_path = Path(sys.argv[2])
adapter_ratio = float(sys.argv[3])
payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
if split_path.is_file():
    payload['split_manifest_sha256'] = hashlib.sha256(
        split_path.read_bytes()).hexdigest()
payload['adapter_dim_ratio'] = adapter_ratio
torch.save(payload, ckpt_path)
print(f'checkpoint ready: {ckpt_path}')
PY

echo "checkpoint: $WORKDIR/best_model.pth"
echo "next: python scripts/run_baseline_test.py --method sacm_official --no-resume"
