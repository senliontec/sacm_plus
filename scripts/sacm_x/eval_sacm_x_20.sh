#!/usr/bin/env bash
# Evaluate the saved best SACM-X checkpoint on the 20 frozen-protocol test sets.
# No retraining.
#
# Usage:
#   bash scripts/sacm_x/eval_sacm_x_20.sh
#   GPU=2 bash scripts/sacm_x/eval_sacm_x_20.sh
#   DATASETS="DRIVE WIRE" bash scripts/sacm_x/eval_sacm_x_20.sh
#
# Environment:
#   GPU (default 3), OUTPUT_ROOT, PRESET (default full),
#   SAM_CKPT, DATASETS, USE_WANDB, WANDB_MODE,
#   SKIP_EVAL=1 (only aggregate + report existing results)
set -uo pipefail
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

GPU="${GPU:-3}"
PRESET="${PRESET:-full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/sacm_x_official_200}"
CKPT="$OUTPUT_ROOT/$PRESET/best_model.pth"
SAM_CKPT="${SAM_CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
DATASETS="${DATASETS:-}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
DETERMINISTIC="${DETERMINISTIC:-true}"
DETERMINISTIC_STRICT="${DETERMINISTIC_STRICT:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot}"
EVAL_INPUT_MODE="${EVAL_INPUT_MODE:-aspect_pad}"
USE_TTA="${USE_TTA:-false}"
EMPTY_GUARD="${EMPTY_GUARD:-true}"
SELECTION="${SELECTION:-}"
FIXED_HEAD="${FIXED_HEAD:-}"
QUALITY_SMALL_COMPONENT_WEIGHT="${QUALITY_SMALL_COMPONENT_WEIGHT:-0.5}"
QUALITY_COMPONENT_WEIGHT="${QUALITY_COMPONENT_WEIGHT:-0.05}"
AUDIT_HEADS="${AUDIT_HEADS:-false}"
AUDIT_HEADS_FULL="${AUDIT_HEADS_FULL:-false}"
SAVE_CANDIDATE_MASKS="${SAVE_CANDIDATE_MASKS:-false}"
COMPONENT_METRICS="${COMPONENT_METRICS:-false}"
COMPONENT_PRUNE_MODE="${COMPONENT_PRUNE_MODE:-off}"
COMPONENT_PRUNE_MIN_AREA="${COMPONENT_PRUNE_MIN_AREA:-0}"
COMPONENT_PRUNE_MIN_SKELETON="${COMPONENT_PRUNE_MIN_SKELETON:-0}"
COMPONENT_PRUNE_REL_AREA="${COMPONENT_PRUNE_REL_AREA:-0.0}"
COMPONENT_PRUNE_REL_SKELETON="${COMPONENT_PRUNE_REL_SKELETON:-0.0}"
COMPONENT_PRUNE_KEEP_LARGEST="${COMPONENT_PRUNE_KEEP_LARGEST:-1}"
AUDIT_THRESHOLDS="${AUDIT_THRESHOLDS:-}"
AUDIT_HEAD_THRESHOLDS="${AUDIT_HEAD_THRESHOLDS:-}"
DEEP_CLOSING_CKPT="${DEEP_CLOSING_CKPT:-}"
DEEP_CLOSING_THRESHOLD="${DEEP_CLOSING_THRESHOLD:-}"
SKIP_EVAL="${SKIP_EVAL:-false}"
export WANDB_MODE
source scripts/sacm_x/wandb_env.sh

if [ ! -f "$CKPT" ]; then
  echo "ERROR: missing checkpoint: $CKPT" >&2
  echo "Train first or set OUTPUT_ROOT/CKPT correctly." >&2
  exit 2
fi
if [ ! -f "$TRAIN_ROOT/protocol_manifest.csv" ]; then
  echo "ERROR: missing protocol: $TRAIN_ROOT/protocol_manifest.csv" >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || used="?"
  echo "GPU $GPU memory used: ${used:-?} MiB"
  if [[ "${used:-0}" =~ ^[0-9]+$ ]] && [ "$used" -gt 2000 ]; then
    echo "ERROR: GPU $GPU already has ${used} MiB in use; refusing to start evaluation." >&2
    echo "Free it or run on another card, e.g. GPU=2 bash scripts/sacm_x/eval_sacm_x_20.sh" >&2
    nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null || true
    exit 3
  fi
fi

# Always take test roots from the same split root as training.  A seed-specific
# TRAIN_ROOT must not silently fall back to data/sacm_3shot's test sets.
SPECS="$("$PYTHON" scripts/pipeline/protocol_specs.py \
  --protocol "$TRAIN_ROOT/protocol_manifest.csv" \
  --datasets "$DATASETS")"
read -r -a spec_array <<< "$SPECS"

echo "checkpoint=$CKPT"
echo "datasets=${#spec_array[@]}  output=$OUTPUT_ROOT"
if [ -n "$DEEP_CLOSING_CKPT" ]; then
  if [ ! -f "$DEEP_CLOSING_CKPT" ]; then
    echo "ERROR: missing Deep Closing checkpoint: $DEEP_CLOSING_CKPT" >&2
    exit 2
  fi
  echo "deep_closing=$DEEP_CLOSING_CKPT"
fi

if [ "$SKIP_EVAL" = "true" ] || [ "$SKIP_EVAL" = "1" ]; then
  echo "SKIP_EVAL=$SKIP_EVAL: using existing per-dataset results"
  rc=0
else
  echo "no retraining; evaluating with --skip_train"
  echo "eval_input_mode=$EVAL_INPUT_MODE tta=$USE_TTA empty_guard=$EMPTY_GUARD"
  EXTRA_EVAL="--use_wandb $USE_WANDB --no-fov"
  if [ -n "$DEEP_CLOSING_CKPT" ]; then
    EXTRA_EVAL+=" --deep_closing_checkpoint $DEEP_CLOSING_CKPT"
    if [ -n "$DEEP_CLOSING_THRESHOLD" ]; then
      EXTRA_EVAL+=" --deep_closing_threshold $DEEP_CLOSING_THRESHOLD"
    fi
  fi
  EXTRA_EVAL+=" --quality_small_component_weight $QUALITY_SMALL_COMPONENT_WEIGHT"
  EXTRA_EVAL+=" --quality_component_weight $QUALITY_COMPONENT_WEIGHT"
  if [ "$COMPONENT_METRICS" = "true" ] || [ "$COMPONENT_METRICS" = "1" ]; then
    EXTRA_EVAL+=" --component_metrics"
  fi
  if [ "$COMPONENT_PRUNE_MODE" != "off" ]; then
    EXTRA_EVAL+=" --component_prune_mode $COMPONENT_PRUNE_MODE"
    EXTRA_EVAL+=" --component_prune_min_area $COMPONENT_PRUNE_MIN_AREA"
    EXTRA_EVAL+=" --component_prune_min_skeleton $COMPONENT_PRUNE_MIN_SKELETON"
    EXTRA_EVAL+=" --component_prune_rel_area $COMPONENT_PRUNE_REL_AREA"
    EXTRA_EVAL+=" --component_prune_rel_skeleton $COMPONENT_PRUNE_REL_SKELETON"
    EXTRA_EVAL+=" --component_prune_keep_largest $COMPONENT_PRUNE_KEEP_LARGEST"
  fi
  if [ "$AUDIT_HEADS" = "true" ] || [ "$AUDIT_HEADS" = "1" ]; then
    EXTRA_EVAL+=" --audit_heads"
  fi
  if [ "$AUDIT_HEADS_FULL" = "true" ] || [ "$AUDIT_HEADS_FULL" = "1" ]; then
    EXTRA_EVAL+=" --audit_heads_full"
  fi
  if [ "$SAVE_CANDIDATE_MASKS" = "true" ] || [ "$SAVE_CANDIDATE_MASKS" = "1" ]; then
    EXTRA_EVAL+=" --save_candidate_masks"
  fi
  if [ -n "$AUDIT_THRESHOLDS" ]; then
    EXTRA_EVAL+=" --audit_thresholds $AUDIT_THRESHOLDS"
  fi
  if [ -n "$AUDIT_HEAD_THRESHOLDS" ]; then
    EXTRA_EVAL+=" --audit_head_thresholds $AUDIT_HEAD_THRESHOLDS"
  fi
  EXTRA_RUN=(--eval_input_mode "$EVAL_INPUT_MODE"
    --deterministic "$DETERMINISTIC"
    --deterministic_strict "$DETERMINISTIC_STRICT"
    --num_workers "$NUM_WORKERS")
  if [ -n "$SELECTION" ]; then
    EXTRA_RUN+=(--selection "$SELECTION")
  fi
  if [ -n "$FIXED_HEAD" ]; then
    EXTRA_RUN+=(--fixed_head "$FIXED_HEAD")
  fi
  if [ "$EMPTY_GUARD" = "true" ] || [ "$EMPTY_GUARD" = "1" ]; then
    EXTRA_RUN+=(--empty_guard)
  fi
  if [ "$USE_TTA" = "true" ] || [ "$USE_TTA" = "1" ]; then
    EXTRA_RUN+=(--tta)
  fi
  "$PYTHON" scripts/sacm_x/run_experiments.py \
    --gpus "$GPU" \
    --checkpoint "$SAM_CKPT" \
    --train_root "$TRAIN_ROOT" \
    --test_datasets "${spec_array[@]}" \
    --presets "$PRESET" \
    --output_root "$OUTPUT_ROOT" \
    --skip_train \
    "${EXTRA_RUN[@]}" \
    --extra_eval_args "$EXTRA_EVAL"
  rc=$?
fi

echo
echo "== aggregate (partial results are kept if any dataset failed) =="
agg_rc=0
"$PYTHON" scripts/sacm_x/aggregate_results.py \
  --root "$OUTPUT_ROOT" --out "$OUTPUT_ROOT/tables" || agg_rc=$?

echo
echo "== final report =="
report_rc=0
"$PYTHON" scripts/sacm_x/make_final_report.py \
  --tables-dir "$OUTPUT_ROOT/tables" \
  --preset "$PRESET" \
  --checkpoint "$CKPT" || report_rc=$?

if [ "$rc" -ne 0 ] || [ "$agg_rc" -ne 0 ] || [ "$report_rc" -ne 0 ]; then
  echo "ERROR: eval/aggregate/report failed (eval=$rc aggregate=$agg_rc report=$report_rc)" >&2
  exit 1
fi
exit 0
