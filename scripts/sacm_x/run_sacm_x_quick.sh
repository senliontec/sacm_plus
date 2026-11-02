#!/usr/bin/env bash
# Quick same-protocol test for SACM-X's own model.
#
# 18-image train_root, all 20 frozen-protocol test sets, ONE preset,
# no TTA, W&B on, baseline-compatible validation, then aggregate the 4
# paper metrics.
#
# Usage:
#   GPU=1 PRESET=full EPOCHS=400 bash scripts/sacm_x/run_sacm_x_quick.sh
#   GPU=1 PRESET=sacm EPOCHS=20 MONITOR=official bash scripts/sacm_x/run_sacm_x_quick.sh
#   GPU=1 PRESET=full EPOCHS=5 DATASETS="CHASEDB1 WIRE" \
#     bash scripts/sacm_x/run_sacm_x_quick.sh
#
# Environment:
#   PYTHON, GPU, PRESET, RUN_NAME, EPOCHS, VAL_INTERVAL, SEED, DATASETS, CKPT,
#   TRAIN_ROOT, OUTPUT_ROOT, MONITOR, USE_WANDB, WANDB_MODE, WANDB_NAME,
#   INIT_STAGE2_FROM_STAGE1, SKIP_TEST, SCNP, SCNP_WINDOW,
#   SCNP_ORIGINAL_WEIGHT
#
# MONITOR=baseline (default): every-epoch macro Dice/IoU/clDice/HD95 and
#   per-dataset W&B keys, exactly like src/baseline_zoo/runtime.py;
#   checkpoint selection uses macro Dice; no FOV.
# MONITOR=official: official SACM-style loss/F1/LR terminal lines, no FOV,
#   and only loss/F1 during training validation.
# MONITOR=sacm: fast validation but still prints the rich wide table (legacy).
# MONITOR=full: re-enable the full validation metric suite (Dice/IoU/clDice/
#   HD95 + AUC/Betti/Topograph engines).
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
REPO="$(pwd)"

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
RUN_NAME="${RUN_NAME:-${PRESET}}"
source scripts/sacm_x/wandb_env.sh
EPOCHS="${EPOCHS:-400}"
VAL_INTERVAL="${VAL_INTERVAL:-}"
SEED="${SEED:-0}"
DATASETS="${DATASETS:-}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/sacm_x/${RUN_NAME}_seed${SEED}/train}"
if [[ "$OUTPUT_ROOT" != *"seed_${SEED}"* ]]; then
  echo "WARNING: OUTPUT_ROOT does not contain seed_${SEED}; different seeds can collide." >&2
fi
MONITOR="${MONITOR:-baseline}"
INIT_STAGE2_FROM_STAGE1="${INIT_STAGE2_FROM_STAGE1:-true}"
INPUT_MODE="${INPUT_MODE:-aspect_pad}"
EVAL_INPUT_MODE="${EVAL_INPUT_MODE:-aspect_pad}"
FG_CROP_PROB="${FG_CROP_PROB:-0.5}"
BOUNDARY_LOSS_WEIGHT="${BOUNDARY_LOSS_WEIGHT:-0.05}"
BOUNDARY_LOSS_RESOLUTION="${BOUNDARY_LOSS_RESOLUTION:-512}"
BOUNDARY_LOSS_WARMUP="${BOUNDARY_LOSS_WARMUP:-10}"
BOUNDARY_LOSS_RAMP="${BOUNDARY_LOSS_RAMP:-5}"
USE_TTA="${USE_TTA:-false}"
EMPTY_GUARD="${EMPTY_GUARD:-true}"
SCNP="${SCNP:-false}"
SCNP_WINDOW="${SCNP_WINDOW:-3}"
SCNP_ORIGINAL_WEIGHT="${SCNP_ORIGINAL_WEIGHT:-0.0}"
DEEP_CLOSING_CKPT="${DEEP_CLOSING_CKPT:-}"
DEEP_CLOSING_THRESHOLD="${DEEP_CLOSING_THRESHOLD:-}"
SKIP_TEST="${SKIP_TEST:-false}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
DETERMINISTIC="${DETERMINISTIC:-true}"
DETERMINISTIC_STRICT="${DETERMINISTIC_STRICT:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
export WANDB_MODE
if [ "$USE_WANDB" = "true" ] && [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled (mode=$WANDB_MODE) but no credentials were found." >&2
    echo "Set WANDB_API_KEY or ~/.netrc, or use WANDB_MODE=offline / USE_WANDB=false." >&2
    exit 1
  fi
fi
case "$MONITOR" in
  baseline) TRAIN_EXTRA=(--use_wandb "$USE_WANDB" --fast_val false
                --val_auc false --val_betti_matching false --val_topograph false
                --log_style baseline --no-fov)
              EVAL_EXTRA="--use_wandb $USE_WANDB --no-fov"
              [ -n "$VAL_INTERVAL" ] || VAL_INTERVAL=1 ;;
  official) TRAIN_EXTRA=(--use_wandb "$USE_WANDB" --fast_val true
              --val_auc false --val_betti_matching false --val_topograph false
              --log_style official --no-fov)
            EVAL_EXTRA="--use_wandb $USE_WANDB --no-fov"
            [ -n "$VAL_INTERVAL" ] || VAL_INTERVAL=10 ;;
  sacm) TRAIN_EXTRA=(--use_wandb "$USE_WANDB" --fast_val true
          --val_auc false --val_betti_matching false --val_topograph false)
        EVAL_EXTRA="--use_wandb $USE_WANDB"
        [ -n "$VAL_INTERVAL" ] || VAL_INTERVAL=10 ;;
  full) TRAIN_EXTRA=(--use_wandb "$USE_WANDB")
        EVAL_EXTRA="--use_wandb $USE_WANDB"
        [ -n "$VAL_INTERVAL" ] || VAL_INTERVAL=10 ;;
  *) echo "ERROR: MONITOR must be 'baseline', 'official', 'sacm', or 'full', got '$MONITOR'" >&2
     exit 2 ;;
esac
TRAIN_EXTRA+=(--wandb_name "${WANDB_NAME:-sacm_x_${RUN_NAME}_seed${SEED}}")

TRAIN_EXTRA+=(--scnp "$SCNP" --scnp_window "$SCNP_WINDOW" \
              --scnp_original_weight "$SCNP_ORIGINAL_WEIGHT")

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

echo "seed=$SEED preset=$PRESET epochs=$EPOCHS gpu=$GPU monitor=$MONITOR val_interval=$VAL_INTERVAL init_stage2_from_stage1=$INIT_STAGE2_FROM_STAGE1 input_mode=$INPUT_MODE eval_input_mode=$EVAL_INPUT_MODE fg_crop_prob=$FG_CROP_PROB boundary_weight=$BOUNDARY_LOSS_WEIGHT tta=$USE_TTA empty_guard=$EMPTY_GUARD scnp=$SCNP($SCNP_WINDOW) deep_closing=${DEEP_CLOSING_CKPT:-none} skip_test=$SKIP_TEST"
echo "datasets=${#spec_array[@]}  output=$OUTPUT_ROOT"
RUN_EXTRA=()
if [ "$SKIP_TEST" = "true" ]; then
  RUN_EXTRA+=(--skip_test)
fi
if [ "$EMPTY_GUARD" = "true" ] || [ "$EMPTY_GUARD" = "1" ]; then
  RUN_EXTRA+=(--empty_guard)
fi
if [ "$USE_TTA" = "true" ] || [ "$USE_TTA" = "1" ]; then
  RUN_EXTRA+=(--tta)
fi
"$PYTHON" scripts/sacm_x/run_experiments.py \
  --gpus "$GPU" \
  --checkpoint "$CKPT" \
  --train_root "$TRAIN_ROOT" \
  --test_datasets "${spec_array[@]}" \
  --presets "$PRESET" \
  --output_root "$OUTPUT_ROOT" \
  --epochs "$EPOCHS" --val_interval "$VAL_INTERVAL" --seed "$SEED" \
  --deterministic "$DETERMINISTIC" \
  --deterministic_strict "$DETERMINISTIC_STRICT" \
  --num_workers "$NUM_WORKERS" \
  --train_input_mode "$INPUT_MODE" --eval_input_mode "$EVAL_INPUT_MODE" \
  --foreground_crop_prob "$FG_CROP_PROB" \
  --boundary_loss_weight "$BOUNDARY_LOSS_WEIGHT" \
  --boundary_loss_resolution "$BOUNDARY_LOSS_RESOLUTION" \
  --boundary_loss_warmup "$BOUNDARY_LOSS_WARMUP" \
  --boundary_loss_ramp "$BOUNDARY_LOSS_RAMP" \
  --extra_train_args "${TRAIN_EXTRA[*]}" \
  --extra_eval_args "$EVAL_EXTRA" \
  ${RUN_EXTRA[@]+"${RUN_EXTRA[@]}"}

echo
echo "== artifacts =="
echo "best checkpoint: $OUTPUT_ROOT/$PRESET/best_model.pth"
echo "training log:    $OUTPUT_ROOT/$PRESET/run.log"
echo "run info:        $OUTPUT_ROOT/$PRESET/run_info.json"
echo "metrics history: $OUTPUT_ROOT/$PRESET/metrics_history.json"
echo "manifest:        $OUTPUT_ROOT/run_manifest.json"
echo "jobs:            $OUTPUT_ROOT/jobs.csv"

if [ "$SKIP_TEST" = "true" ]; then
  echo "== test/eval skipped (SKIP_TEST=true) =="
  exit 0
fi

echo "eval tables:     $OUTPUT_ROOT/tables"

echo "== aggregate =="
AGG_ARGS=(--root "$OUTPUT_ROOT" --out "$OUTPUT_ROOT/tables"
  --preset "$PRESET")
if [ -n "$DATASETS" ]; then
  # shellcheck disable=SC2206
  AGG_ARGS+=(--datasets $DATASETS)
fi
"$PYTHON" scripts/sacm_x/aggregate_results.py "${AGG_ARGS[@]}"

echo "== macro =="
cat "$OUTPUT_ROOT/tables/macro_metrics.csv"

echo "== final report =="
"$PYTHON" scripts/sacm_x/make_final_report.py \
  --tables-dir "$OUTPUT_ROOT/tables" \
  --preset "$PRESET" \
  --checkpoint "$OUTPUT_ROOT/$PRESET/best_model.pth"
