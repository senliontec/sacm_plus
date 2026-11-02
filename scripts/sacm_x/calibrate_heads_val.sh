#!/usr/bin/env bash
# One-click head x threshold calibration on the few-shot val/support split.
#
# It evaluates every Stage-2 head at every requested logit threshold and
# writes:
#   <CALIB_ROOT>/<split>/head_threshold_audit.csv
#   <CALIB_ROOT>/<split>/head_threshold_summary.csv
#   <CALIB_ROOT>/<split>/head_threshold_report.txt
#
# The default split is `val` (6 seen images, one per seen dataset).  This is
# small: use the report only as evidence, not as a per-dataset oracle.  The
# `train` split has 18 support images; it is included for comparison but was
# seen during training, so it is optimistic.  NEVER pass a test split here.
#
# Usage:
#   GPU=7 bash scripts/sacm_x/calibrate_heads_val.sh
#   GPU=7 PRESET=full OUTPUT_ROOT=results/paper/sacm_x/seed_0 \
#     bash scripts/sacm_x/calibrate_heads_val.sh
#   GPU=7 SPLITS="val train" bash scripts/sacm_x/calibrate_heads_val.sh
#
# Environment:
#   PYTHON, GPU, PRESET, OUTPUT_ROOT, CKPT, TRAIN_ROOT, SPLITS, CALIB_ROOT,
#   EVAL_INPUT_MODE (default: resize, matching the current audit),
#   SELECTION (default: iou), EMPTY_GUARD, THRESHOLDS, USE_WANDB.
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
OUTPUT_ROOT="${OUTPUT_ROOT:-results/paper/sacm_x/seed_0}"
CKPT="${CKPT:-$OUTPUT_ROOT/$PRESET/best_model.pth}"
if [ -z "${TRAIN_ROOT:-}" ]; then
  for candidate in data/sacm_3shot_seed0 data/sacm_3shot; do
    if [ -f "$candidate/protocol_manifest.csv" ]; then
      TRAIN_ROOT="$candidate"
      break
    fi
  done
fi
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot_seed0}"
SPLITS="${SPLITS:-val}"
CALIB_ROOT="${CALIB_ROOT:-$OUTPUT_ROOT/calibration}"
EVAL_INPUT_MODE="${EVAL_INPUT_MODE:-resize}"
SELECTION="${SELECTION:-iou}"
EMPTY_GUARD="${EMPTY_GUARD:-true}"
THRESHOLDS="${THRESHOLDS:--2 -1 0 1 2}"
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
  OUT_DIR="$CALIB_ROOT/$split"
  if [ ! -d "$DATA_ROOT/images" ] || [ ! -d "$DATA_ROOT/masks" ]; then
    echo "ERROR: $DATA_ROOT must contain images/ and masks/" >&2
    exit 2
  fi
  mkdir -p "$OUT_DIR"
  if [ "$split" = "train" ]; then
    echo "WARNING: '$split' support images were used for training; calibration is optimistic."
  fi
  echo "== $split: checkpoint=$CKPT data=$DATA_ROOT out=$OUT_DIR =="

  EXTRA=()
  if [ "$EMPTY_GUARD" = "true" ] || [ "$EMPTY_GUARD" = "1" ]; then
    EXTRA+=(--empty_guard)
  fi

  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/eval/evaluate.py \
    --preset "$PRESET" \
    --data_root "$DATA_ROOT" \
    --trained_weights "$CKPT" \
    --output_dir "$OUT_DIR" \
    --selection "$SELECTION" \
    --pred_threshold 0 \
    --eval_input_mode "$EVAL_INPUT_MODE" \
    --no-fov \
    --audit_heads \
    --audit_head_thresholds $THRESHOLDS \
    --component_metrics \
    --deterministic "$DETERMINISTIC" \
    --deterministic_strict "$DETERMINISTIC_STRICT" \
    --num_workers "$NUM_WORKERS" \
    --use_wandb "$USE_WANDB" \
    ${EXTRA[@]+"${EXTRA[@]}"}

  "$PYTHON" scripts/sacm_x/aggregate_head_threshold.py \
    --audit "$OUT_DIR/head_threshold_audit.csv" \
    --protocol "$TRAIN_ROOT/protocol_manifest.csv" \
    --split "$split" \
    --out "$OUT_DIR/head_threshold_summary.csv" \
    | tee "$OUT_DIR/head_threshold_report.txt"
done

echo
echo "Reports:"
for split in $SPLITS; do
  echo "  $CALIB_ROOT/$split/head_threshold_report.txt"
done
