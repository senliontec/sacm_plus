#!/usr/bin/env bash
# Quick same-protocol test for SACM-X's own model.
#
# 18-image train_root, all 20 frozen-protocol test sets, ONE preset,
# no TTA, W&B on, full validation, then aggregate the 4 paper metrics.
#
# Usage:
#   GPU=1 PRESET=full EPOCHS=200 bash scripts/sacm_x/run_sacm_x_quick.sh
#   GPU=1 PRESET=sacm EPOCHS=20 bash scripts/sacm_x/run_sacm_x_quick.sh
#   GPU=1 PRESET=full EPOCHS=5 DATASETS="CHASEDB1 WIRE" \
#     bash scripts/sacm_x/run_sacm_x_quick.sh
#
# Environment:
#   PYTHON, GPU, MODEL, PRESET, EPOCHS, VAL_INTERVAL, SEED, DATASETS, CKPT,
#   TRAIN_ROOT, OUTPUT_ROOT, MONITOR, USE_WANDB, WANDB_MODE,
#   INIT_STAGE2_FROM_STAGE1, SKIP_TEST, SKIP_AGGREGATE, SCNP, SCNP_WINDOW,
#   SCNP_ORIGINAL_WEIGHT, WANDB_GROUP (fallback: WANDB_RUN_GROUP)
#
# MODEL defaults to sam_l (the ViT-L checkpoint). For a local ViT-B smoke,
# set MODEL=sam_b and CKPT=checkpoints/sam_vit_b_01ec64.pth.
#
# MONITOR=official (default): baseline-format compact table and full
#   validation metrics (Dice/IoU/clDice/HD95), no FOV in validation or test.
# MONITOR=sacm: fast validation (loss/F1 only; table metric cells show —).
# MONITOR=full: full validation metric suite (AUC/Betti/Topograph engines,
#   plus FOV masking).
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
REPO="$(pwd)"

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
source scripts/sacm_x/wandb_env.sh
EPOCHS="${EPOCHS:-200}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
SEED="${SEED:-42}"
DATASETS="${DATASETS:-}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/sacm_x_quick}"
MONITOR="${MONITOR:-official}"
INIT_STAGE2_FROM_STAGE1="${INIT_STAGE2_FROM_STAGE1:-false}"
SCNP="${SCNP:-false}"
SCNP_WINDOW="${SCNP_WINDOW:-3}"
SCNP_ORIGINAL_WEIGHT="${SCNP_ORIGINAL_WEIGHT:-0.0}"
DEEP_CLOSING_CKPT="${DEEP_CLOSING_CKPT:-}"
DEEP_CLOSING_THRESHOLD="${DEEP_CLOSING_THRESHOLD:-}"
SKIP_TEST="${SKIP_TEST:-false}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-false}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-${WANDB_RUN_GROUP:-}}"
export WANDB_MODE
if [ "$USE_WANDB" = "true" ] && [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled (mode=$WANDB_MODE) but no credentials were found." >&2
    echo "Set WANDB_API_KEY or ~/.netrc, or use WANDB_MODE=offline / USE_WANDB=false." >&2
    exit 1
  fi
fi
case "$MONITOR" in
  official) TRAIN_EXTRA=(--use_wandb "$USE_WANDB" --fast_val false
              --val_auc false --val_betti_matching false --val_topograph false
              --log_style table --no-fov)
            EVAL_EXTRA="--use_wandb $USE_WANDB --no-fov" ;;
  sacm) TRAIN_EXTRA=(--use_wandb "$USE_WANDB" --fast_val true
          --val_auc false --val_betti_matching false --val_topograph false)
        EVAL_EXTRA="--use_wandb $USE_WANDB" ;;
  full) TRAIN_EXTRA=(--use_wandb "$USE_WANDB")
        EVAL_EXTRA="--use_wandb $USE_WANDB" ;;
  *) echo "ERROR: MONITOR must be 'official', 'sacm', or 'full', got '$MONITOR'" >&2
     exit 2 ;;
esac

TRAIN_EXTRA+=(--model "$MODEL" \
              --scnp "$SCNP" --scnp_window "$SCNP_WINDOW" \
              --scnp_original_weight "$SCNP_ORIGINAL_WEIGHT")
if [ -n "$WANDB_GROUP" ]; then
  TRAIN_EXTRA+=(--wandb_group "$WANDB_GROUP")
fi
EVAL_EXTRA+=" --model $MODEL"

if [ -n "$DEEP_CLOSING_CKPT" ]; then
  if [ ! -f "$DEEP_CLOSING_CKPT" ]; then
    echo "ERROR: missing Deep Closing checkpoint: $DEEP_CLOSING_CKPT" >&2
    exit 2
  fi
  EVAL_EXTRA+=" --deep_closing_checkpoint $DEEP_CLOSING_CKPT"
  if [ -n "$DEEP_CLOSING_THRESHOLD" ]; then
    EVAL_EXTRA+=" --deep_closing_threshold $DEEP_CLOSING_THRESHOLD"
  fi
fi

if [ "$INIT_STAGE2_FROM_STAGE1" = "true" ]; then
  TRAIN_EXTRA+=(--init_stage2_from_stage1 true)
fi

if [ ! -f "$CKPT" ]; then
  echo "ERROR: missing SAM ViT-L checkpoint: $CKPT" >&2
  exit 2
fi
if [ ! -f "$TRAIN_ROOT/protocol_manifest.csv" ]; then
  echo "ERROR: missing protocol: $TRAIN_ROOT/protocol_manifest.csv" >&2
  exit 2
fi

# Always take test roots from the same split root as training.  A seed-specific
# TRAIN_ROOT must not silently fall back to data/sacm_3shot's test sets.
SPECS="$("$PYTHON" scripts/pipeline/protocol_specs.py \
  --protocol "$TRAIN_ROOT/protocol_manifest.csv" \
  --datasets "$DATASETS")"
read -r -a spec_array <<< "$SPECS"

echo "model=$MODEL preset=$PRESET epochs=$EPOCHS gpu=$GPU monitor=$MONITOR init_stage2_from_stage1=$INIT_STAGE2_FROM_STAGE1 scnp=$SCNP($SCNP_WINDOW) deep_closing=${DEEP_CLOSING_CKPT:-none} skip_test=$SKIP_TEST"
echo "datasets=${#spec_array[@]}  output=$OUTPUT_ROOT"
RUN_EXTRA=()
if [ "$SKIP_TEST" = "true" ]; then
  RUN_EXTRA+=(--skip_test)
fi
"$PYTHON" scripts/sacm_x/run_experiments.py \
  --gpus "$GPU" \
  --checkpoint "$CKPT" \
  --train_root "$TRAIN_ROOT" \
  --test_datasets "${spec_array[@]}" \
  --presets "$PRESET" \
  --output_root "$OUTPUT_ROOT" \
  --epochs "$EPOCHS" --val_interval "$VAL_INTERVAL" --seed "$SEED" \
  --extra_train_args "${TRAIN_EXTRA[*]}" \
  --extra_eval_args "$EVAL_EXTRA" \
  ${RUN_EXTRA[@]+"${RUN_EXTRA[@]}"}

if [ "$SKIP_TEST" = "true" ]; then
  echo "== test/eval skipped (SKIP_TEST=true) =="
  echo "checkpoint/log dir: $OUTPUT_ROOT/$PRESET"
  exit 0
fi

if [ "$SKIP_AGGREGATE" = "true" ]; then
  echo "== aggregate skipped (SKIP_AGGREGATE=true) =="
  exit 0
fi

echo "== aggregate =="
"$PYTHON" scripts/sacm_x/aggregate_results.py \
  --root "$OUTPUT_ROOT" --out "$OUTPUT_ROOT/tables"

echo "== macro =="
cat "$OUTPUT_ROOT/tables/macro_metrics.csv"
