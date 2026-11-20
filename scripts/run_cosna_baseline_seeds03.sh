#!/usr/bin/env bash
# One-click CoSNA baseline runner.
#
# MODE=repro (default): copy the four canonical CoSNA checkpoints into the
#   baseline run roots, patch only the copied checkpoint's adapter tag, and
#   evaluate all 20 datasets with the standard baseline pipeline.
# MODE=train: initialize each seed from its BCTD champion checkpoint, fit the
#   TCSU head for EPOCHS_ADD epochs, then evaluate.
#
# Usage:
#   bash scripts/run_cosna_baseline_seeds03.sh
#   MODE=train bash scripts/run_cosna_baseline_seeds03.sh
#   SEEDS="0" GPUS="0" bash scripts/run_cosna_baseline_seeds03.sh
#
# Env:
#   SEEDS="0 1 2 3", GPUS="0,1,2,3", EPOCHS_ADD=20
#   RUN_ROOT=results/runs/cosna/default
#   REPRO_ROOT=results/runs/bctd/tcsu20
#   BCTD_ROOT=results/runs/topostrand/bctd200
#   SEED0_SOURCE=results/runs/topostrand/pilot200_bnd005_h2/seed_0/best_model_fixed3.pth
#   FORCE=0, SKIP_TRAIN=0, SKIP_EVAL=0, USE_WANDB=1
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SEEDS_STR="${SEEDS:-0 1 2 3}"
SEEDS_STR="${SEEDS_STR//,/ }"
GPU_STR="${GPUS:-0,1,2,3}"
MODE="${MODE:-repro}"
RUN_ROOT="${RUN_ROOT:-results/runs/cosna/default}"
REPRO_ROOT="${REPRO_ROOT:-results/runs/bctd/tcsu20}"
BCTD_ROOT="${BCTD_ROOT:-results/runs/topostrand/bctd200}"
SEED0_SOURCE="${SEED0_SOURCE:-results/runs/topostrand/pilot200_bnd005_h2/seed_0/best_model_fixed3.pth}"
EPOCHS_ADD="${EPOCHS_ADD:-20}"
FORCE="${FORCE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-$REPO/results/.tmp}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$TMPDIR" "$REPO/results/logs"

if [ "$USE_WANDB" != 0 ]; then
  source "$REPO/scripts/wandb_env.sh"
fi

read -r -a SEEDS <<< "$SEEDS_STR"
read -r -a GPUS <<< "${GPU_STR//,/ }"
if [ "${#SEEDS[@]}" -eq 0 ] || [ "${#GPUS[@]}" -eq 0 ]; then
  echo "ERROR: empty SEEDS/GPUS" >&2
  exit 2
fi

run_one_seed() {
  local seed="$1" gpu="$2"
  local data="$REPO/data/sacm_3shot_seed${seed}"
  local work="$REPO/$RUN_ROOT/seed_${seed}"
  local log="$REPO/results/logs/cosna_baseline_seed${seed}.log"
  local source
  if [ "$seed" = 0 ]; then
    source="$REPO/$SEED0_SOURCE"
  else
    source="$REPO/$BCTD_ROOT/seed_${seed}/best_model_fixed3.pth"
  fi
  mkdir -p "$work"
  : > "$log"
  echo "mode=$MODE seed=$seed gpu=$gpu work=$work" >> "$log"

  if [ "$MODE" = repro ]; then
    if [ ! -f "$work/best_model.pth" ] || [ "$FORCE" = 1 ]; then
      local repro_source="$REPO/$REPRO_ROOT/seed_${seed}/best_model.pth"
      if [ ! -f "$repro_source" ]; then
        echo "missing canonical CoSNA checkpoint: $repro_source" >&2
        return 2
      fi
      local prep_cmd=("$PYTHON" -m baseline_zoo.models.cosna.prepare_init
        --source "$repro_source" --workdir "$work" --data-root "$data"
        --epochs-add 0 --force)
      if [ "$DRY_RUN" = 1 ]; then
        printf 'DRY repro-init seed=%s gpu=%s: ' "$seed" "$gpu"
        printf '%q ' "${prep_cmd[@]}"; printf '\n'
      else
        "${prep_cmd[@]}" >> "$log" 2>&1
      fi
    fi
  elif [ "$MODE" = train ]; then
    if [ "$SKIP_TRAIN" != 1 ]; then
      local train_cmd=("$PYTHON" -m baseline_zoo.models.cosna.train
        --data-root "$data" --workdir "$work"
        --source-checkpoint "$source" --epochs-add "$EPOCHS_ADD"
        --seed "$seed" --device "cuda:${gpu}" --use_wandb "$USE_WANDB"
        --sam-checkpoint "$REPO/checkpoints/sam_vit_l_0b3195.pth"
        --lr 1e-3 --cosna-weight 1.0 --cosna-cl 0.1
        --cosna-render 1024 --cosna-coarse 512 --cosna-feature-dim 16
        --cosna-gate 0.10 --cosna-near-radius 4
        --cosna-filter-min-size 80 --cosna-filter-aspect 2
        --cosna-filter-maxdim 1200
        --rcd 1 --as-calibration 1 --close-radius 2
        --tile 1024 --stride 896
        --routing-skel-ratio 1.0 --routing-area-ratio 0.7
        --routing-mode auto --num-workers 0 --save-every 0
        --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY"
        --wandb_group cosna-baseline --wandb_name "cosna-seed${seed}")
      if [ "$DRY_RUN" = 1 ]; then
        printf 'DRY train seed=%s gpu=%s: ' "$seed" "$gpu"
        printf '%q ' "${train_cmd[@]}"; printf '\n'
      else
        "${train_cmd[@]}" >> "$log" 2>&1
      fi
    fi
  else
    echo "ERROR: MODE must be repro or train, got $MODE" >&2
    return 2
  fi

  if [ "$SKIP_EVAL" != 1 ]; then
    local wandb_flag=true
    [ "$USE_WANDB" = 0 ] && wandb_flag=false
    local eval_cmd=("$PYTHON" "$REPO/scripts/pipeline/run_baseline_test.py"
      --method cosna
      --protocol "$data/protocol_manifest.csv"
      --seed "$seed"
      --checkpoint "$work/best_model.pth"
      --data-root "$data"
      --pred-root "$work/predictions"
      --paper-root "$work/eval"
      --device "cuda:${gpu}" --use_wandb "$wandb_flag" --skip-visualize)
    if [ "$DRY_RUN" = 1 ]; then
      printf 'DRY eval seed=%s gpu=%s: ' "$seed" "$gpu"
      printf '%q ' "${eval_cmd[@]}"; printf '\n'
    else
      "${eval_cmd[@]}" >> "$log" 2>&1
    fi
  fi
  echo "DONE seed=$seed mode=$MODE" >> "$log"
}

pids=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  run_one_seed "$seed" "$gpu" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
