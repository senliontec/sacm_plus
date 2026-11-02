#!/usr/bin/env bash
# One-click full-metric per-head audit on the val/support split.
#
# Unlike calibrate_heads_val.sh (quick Dice/IoU/P/R grid), this computes the
# full metric suite for every Stage-2 head: clDice, HD95, ASSD, NSD, etc.
# It answers whether head2/head0 really improve topology/distance metrics
# while Dice is nearly unchanged.
#
# Usage:
#   GPU=7 bash scripts/sacm_x/audit_heads_full_val.sh
#   GPU=7 SPLITS="val train" bash scripts/sacm_x/audit_heads_full_val.sh
#
# Writes:
#   <AUDIT_ROOT>/<split>/head_full_audit.csv
#   <AUDIT_ROOT>/<split>/head_full_summary.csv
#   <AUDIT_ROOT>/<split>/head_full_report.txt
#
# NEVER pass a test split. Use eval_sacm_x_20.sh only for a one-shot
# diagnostic after the deployment policy has been frozen on val/support.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

# Reproducibility: these must be set before Python starts.
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

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
PRESET="${PRESET:-full}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/runs/sacm_x/$PRESET/seed_$SEED}"
CKPT="${CKPT:-$OUTPUT_ROOT/best_model.pth}"
if [ ! -f "$CKPT" ] && [ -f "$OUTPUT_ROOT/$PRESET/best_model.pth" ]; then
  CKPT="$OUTPUT_ROOT/$PRESET/best_model.pth"
fi
if [ -z "${TRAIN_ROOT:-}" ]; then
  for candidate in "data/sacm_3shot_seed$SEED" data/sacm_3shot; do
    if [ -f "$candidate/protocol_manifest.csv" ]; then
      TRAIN_ROOT="$candidate"
      break
    fi
  done
fi
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot_seed$SEED}"
SPLITS="${SPLITS:-val}"
AUDIT_ROOT="${AUDIT_ROOT:-$OUTPUT_ROOT/eval/head_full_audit}"
EVAL_INPUT_MODE="${EVAL_INPUT_MODE:-resize}"
SELECTION="${SELECTION:-iou}"
USE_WANDB="${USE_WANDB:-false}"
DETERMINISTIC="${DETERMINISTIC:-true}"
DETERMINISTIC_STRICT="${DETERMINISTIC_STRICT:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: missing checkpoint: $CKPT" >&2
  exit 2
fi
if [ ! -f "$TRAIN_ROOT/protocol_manifest.csv" ]; then
  echo "ERROR: missing protocol: $TRAIN_ROOT/protocol_manifest.csv" >&2
  exit 2
fi
if [ "$USE_WANDB" = "true" ]; then
  source scripts/sacm_x/wandb_env.sh
fi

for split in $SPLITS; do
  case "$split" in
    val|train) ;;
    *) echo "ERROR: refusing split '$split'; use val or train only" >&2
       exit 2 ;;
  esac
  DATA_ROOT="$TRAIN_ROOT/$split"
  OUT_DIR="$AUDIT_ROOT/$split"
  if [ ! -d "$DATA_ROOT/images" ] || [ ! -d "$DATA_ROOT/masks" ]; then
    echo "ERROR: $DATA_ROOT must contain images/ and masks/" >&2
    exit 2
  fi
  mkdir -p "$OUT_DIR"
  if [ "$split" = "train" ]; then
    echo "WARNING: '$split' support images were used for training; audit is optimistic."
  fi
  echo "== $split full head audit: checkpoint=$CKPT data=$DATA_ROOT out=$OUT_DIR =="

  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/eval/evaluate.py \
    --preset "$PRESET" \
    --data_root "$DATA_ROOT" \
    --trained_weights "$CKPT" \
    --output_dir "$OUT_DIR" \
    --selection "$SELECTION" \
    --pred_threshold 0 \
    --eval_input_mode "$EVAL_INPUT_MODE" \
    --no-fov \
    --audit_heads_full \
    --component_metrics \
    --deterministic "$DETERMINISTIC" \
    --deterministic_strict "$DETERMINISTIC_STRICT" \
    --num_workers "$NUM_WORKERS" \
    --use_wandb "$USE_WANDB"

  "$PYTHON" scripts/sacm_x/aggregate_head_full.py \
    --audit "$OUT_DIR/head_full_audit.csv" \
    --protocol "$TRAIN_ROOT/protocol_manifest.csv" \
    --split "$split" \
    --out "$OUT_DIR/head_full_summary.csv" \
    | tee "$OUT_DIR/head_full_report.txt"
done

echo
echo "Reports:"
for split in $SPLITS; do
  echo "  $AUDIT_ROOT/$split/head_full_report.txt"
done
