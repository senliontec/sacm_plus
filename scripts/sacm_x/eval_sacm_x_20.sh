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
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot}"
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

if [ "$SKIP_EVAL" = "true" ] || [ "$SKIP_EVAL" = "1" ]; then
  echo "SKIP_EVAL=$SKIP_EVAL: using existing per-dataset results"
  rc=0
else
  echo "no retraining; evaluating with --skip_train"
  "$PYTHON" scripts/sacm_x/run_experiments.py \
    --gpus "$GPU" \
    --checkpoint "$SAM_CKPT" \
    --train_root "$TRAIN_ROOT" \
    --test_datasets "${spec_array[@]}" \
    --presets "$PRESET" \
    --output_root "$OUTPUT_ROOT" \
    --skip_train \
    --extra_eval_args "--use_wandb $USE_WANDB --no-fov"
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
