#!/usr/bin/env bash
# 8-GPU recipe matrix for SACM-X (seed 0 by default, 800 epochs).
#
# Each recipe trains ONE preset with a specific recipe override on one GPU.
# By default it only trains (SKIP_TEST=true); set SKIP_TEST=false to also
# evaluate the 20 protocol test sets for every recipe.
#
# Usage:
#   GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/sacm_x/run_recipe_matrix.sh
#   RECIPES="full_default full_noaug" bash scripts/sacm_x/run_recipe_matrix.sh
#   SKIP_TEST=false bash scripts/sacm_x/run_recipe_matrix.sh
#   DRY_RUN=1 bash scripts/sacm_x/run_recipe_matrix.sh   # print commands only
#
# Outputs: results/recipe_matrix/<recipe>/<preset>/{best_model.pth,run.log}
#
# W&B is online by default (self-hosted); the shared wandb_env.sh is sourced
# and credentials are checked before any training starts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi

source scripts/sacm_x/wandb_env.sh
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE
USE_WANDB="${USE_WANDB:-true}"
if [ "$USE_WANDB" = "true" ] && [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled (mode=$WANDB_MODE) but no credentials were found." >&2
    exit 1
  fi
fi

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-800}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/recipe_matrix}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
SKIP_TEST="${SKIP_TEST:-true}"
DATASETS="${DATASETS:-}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "$CKPT" ]; then
  echo "missing SAM checkpoint: $CKPT" >&2
  exit 2
fi
if [ ! -f "$TRAIN_ROOT/split_manifest.csv" ]; then
  echo "missing split manifest: $TRAIN_ROOT/split_manifest.csv" >&2
  exit 2
fi
if [ ! -f "$TRAIN_ROOT/protocol_manifest.csv" ]; then
  echo "missing protocol manifest: $TRAIN_ROOT/protocol_manifest.csv" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

# Keep the training-time validation cheap and identical across recipes.
COMMON_TRAIN_ARGS="--fast_val true --val_auc false --val_betti_matching false --val_topograph false --log_style official --no-fov"

RECIPE_NAMES=(
  sacm_control
  full_default
  full_noaug
  full_lr
  full_noaug_lr
  full_lightloss
  full_stage2off
  stage2_control
)
RECIPE_PRESETS=(sacm full full full full full full stage2)
RECIPE_ARGS=(
  "--no_augment --selection_metric dice"
  "--selection_metric dice"
  "--no_augment --selection_metric dice"
  "--decoder_lr 3e-4 --new_module_lr 3e-4 --selection_metric dice"
  "--no_augment --decoder_lr 3e-4 --new_module_lr 3e-4 --selection_metric dice"
  "--cl_dice_weight 0.1 --iou_loss_weight 0.5 --stage2_supervision allhead --selection_metric dice"
  "--init_stage2_from_stage1 false --selection_metric dice"
  "--no_augment --selection_metric dice"
)

if [ "${RECIPES:-all}" = "all" ]; then
  SELECTED=("${!RECIPE_NAMES[@]}")
else
  SELECTED=()
  for name in $RECIPES; do
    found=0
    for i in "${!RECIPE_NAMES[@]}"; do
      if [ "${RECIPE_NAMES[$i]}" = "$name" ]; then
        SELECTED+=("$i")
        found=1
        break
      fi
    done
    if [ "$found" -eq 0 ]; then
      echo "unknown recipe: $name (known: ${RECIPE_NAMES[*]})" >&2
      exit 2
    fi
  done
fi

SPECS="$("$PYTHON" scripts/pipeline/protocol_specs.py \
  --protocol "$TRAIN_ROOT/protocol_manifest.csv" \
  --datasets "$DATASETS")"
read -r -a SPEC_ARRAY <<< "$SPECS"

echo "SACM-X recipe matrix: seed=$SEED epochs=$EPOCHS train_root=$TRAIN_ROOT"
echo "gpus=${GPUS[*]} recipes=${#SELECTED[@]} skip_test=$SKIP_TEST output=$OUTPUT_ROOT"
echo "W&B mode=$WANDB_MODE use_wandb=$USE_WANDB"

pids=()
names=()
slot=0
for idx in "${SELECTED[@]}"; do
  name="${RECIPE_NAMES[$idx]}"
  preset="${RECIPE_PRESETS[$idx]}"
  extra="${RECIPE_ARGS[$idx]}"
  gpu="${GPUS[$((slot % ${#GPUS[@]}))]}"
  slot=$((slot + 1))
  out="$OUTPUT_ROOT/$name"
  mkdir -p "$out"

  run_extra="$extra $COMMON_TRAIN_ARGS --wandb_name sacm_x_${name}_${preset}"
  if [ "$USE_WANDB" != "true" ]; then
    run_extra="$run_extra --use_wandb false"
  fi

  cmd=( "$PYTHON" scripts/sacm_x/run_experiments.py
        --gpus "$gpu"
        --checkpoint "$CKPT"
        --train_root "$TRAIN_ROOT"
        --test_datasets "${SPEC_ARRAY[@]}"
        --presets "$preset"
        --output_root "$out"
        --epochs "$EPOCHS"
        --val_interval "$VAL_INTERVAL"
        --seed "$SEED"
        --extra_train_args "$run_extra" )
  if [ "$SKIP_TEST" = "true" ]; then
    cmd+=(--skip_test)
  else
    cmd+=(--extra_eval_args "--no-fov")
  fi

  echo "[matrix] $name preset=$preset gpu=$gpu extra='$run_extra'"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run] '
    printf '%q ' "${cmd[@]}"
    echo
    continue
  fi
  "${cmd[@]}" > "$out/driver.log" 2>&1 &
  pids+=("$!")
  names+=("$name")

  if [ "${#pids[@]}" -ge "${#GPUS[@]}" ]; then
    wait "${pids[0]}" || echo "[matrix] ${names[0]} exited nonzero (see $OUTPUT_ROOT/${names[0]}/driver.log)" >&2
    pids=("${pids[@]:1}")
    names=("${names[@]:1}")
  fi
done

for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || echo "[matrix] ${names[$i]} exited nonzero (see $OUTPUT_ROOT/${names[$i]}/driver.log)" >&2
done

echo "== matrix finished =="
for idx in "${SELECTED[@]}"; do
  echo "  ${RECIPE_NAMES[$idx]}: $OUTPUT_ROOT/${RECIPE_NAMES[$idx]}/${RECIPE_PRESETS[$idx]}/best_model.pth"
done
