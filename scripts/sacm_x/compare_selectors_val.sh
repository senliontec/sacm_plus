#!/usr/bin/env bash
# Compare selection policies on the val/support split with full metrics.
#
# Runs: iou, gating (head0), quality (default/strong/rate-only),
# fixed head2, fixed head0.  Every run writes metrics.csv; the aggregator
# reports macro Dice/clDice/HD95/ASSD/NSD and the selected-head counts.
#
# This is a diagnostic, not a test-tuning script.  It only accepts val/train.
#
# Usage:
#   GPU=7 bash scripts/sacm_x/compare_selectors_val.sh
#   GPU=7 SPLITS="val train" bash scripts/sacm_x/compare_selectors_val.sh
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
AUDIT_ROOT="${AUDIT_ROOT:-$OUTPUT_ROOT/eval/selector_audit}"
EVAL_INPUT_MODE="${EVAL_INPUT_MODE:-resize}"
EMPTY_GUARD="${EMPTY_GUARD:-false}"
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

run_eval() {
  local name="$1"
  shift
  local extra=()
  if [ "$EMPTY_GUARD" = "true" ] || [ "$EMPTY_GUARD" = "1" ]; then
    extra+=(--empty_guard)
  fi
  echo "== config=$name =="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" src/eval/evaluate.py \
    --preset "$PRESET" \
    --data_root "$DATA_ROOT" \
    --trained_weights "$CKPT" \
    --output_dir "$OUT_ROOT/$name" \
    --pred_threshold 0 \
    --eval_input_mode "$EVAL_INPUT_MODE" \
    --no-fov \
    --component_metrics \
    --deterministic "$DETERMINISTIC" \
    --deterministic_strict "$DETERMINISTIC_STRICT" \
    --num_workers "$NUM_WORKERS" \
    --use_wandb "$USE_WANDB" \
    "$@" \
    ${extra[@]+"${extra[@]}"}
}

for split in $SPLITS; do
  case "$split" in
    val|train) ;;
    *) echo "ERROR: refusing split '$split'; use val or train only" >&2
       exit 2 ;;
  esac
  DATA_ROOT="$TRAIN_ROOT/$split"
  OUT_ROOT="$AUDIT_ROOT/$split"
  if [ ! -d "$DATA_ROOT/images" ] || [ ! -d "$DATA_ROOT/masks" ]; then
    echo "ERROR: $DATA_ROOT must contain images/ and masks/" >&2
    exit 2
  fi
  mkdir -p "$OUT_ROOT"
  if [ "$split" = "train" ]; then
    echo "WARNING: '$split' support images were used for training; comparison is optimistic."
  fi

  run_eval iou --selection iou
  run_eval gating --selection gating
  run_eval quality_default --selection quality \
    --quality_small_component_weight 0.5 --quality_component_weight 0.05
  run_eval quality_strong --selection quality \
    --quality_small_component_weight 1.0 --quality_component_weight 0.1
  run_eval quality_rate_only --selection quality \
    --quality_small_component_weight 1.0 --quality_component_weight 0.0
  run_eval fixed_head2 --fixed_head 2
  run_eval fixed_head0 --fixed_head 0

  "$PYTHON" scripts/sacm_x/aggregate_selector_val.py \
    --root "$OUT_ROOT" \
    --protocol "$TRAIN_ROOT/protocol_manifest.csv" \
    --split "$split" \
    --out "$OUT_ROOT/selector_summary.csv" \
    | tee "$OUT_ROOT/selector_report.txt"
done

echo
echo "Reports:"
for split in $SPLITS; do
  echo "  $AUDIT_ROOT/$split/selector_report.txt"
done
