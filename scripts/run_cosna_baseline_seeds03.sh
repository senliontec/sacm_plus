#!/usr/bin/env bash
# One-click CoSNA baseline runner.
#
# MODE=repro (default): copy the four canonical CoSNA checkpoints into the
#   baseline run roots, patch only the copied checkpoint's adapter tag, and
#   evaluate all 20 datasets with the standard baseline pipeline.
# MODE=train: initialize each seed from its BCTD champion checkpoint, fit the
#   TCSU head for EPOCHS_ADD epochs (or to EPOCHS_TOTAL), then evaluate.
#
# Usage:
#   bash scripts/run_cosna_baseline_seeds03.sh
#   MODE=train bash scripts/run_cosna_baseline_seeds03.sh
#   SEEDS="0" GPUS="0" bash scripts/run_cosna_baseline_seeds03.sh
#
# Env:
#   SEEDS="0 1 2 3", GPUS="0,1,2,3", EPOCHS_ADD=20
#   EPOCHS_TOTAL=0 (positive = absolute final epoch; overrides EPOCHS_ADD)
#   INIT=scratch|cosna|bctd (train-mode initialization source)
#   COSNA_INIT_ROOT=checkpoints/cosna_init
#   RUN_ROOT=results/runs/cosna/default
#   REPRO_ROOT=results/runs/bctd/tcsu20
#   BCTD_ROOT=results/runs/topostrand/bctd200
#   SEED0_SOURCE=results/runs/topostrand/pilot200_bnd005_h2/seed_0/best_model_fixed3.pth
#   FORCE=0 (MODE=train: FORCE=1 re-initializes from the BCTD source)
#   SKIP_TRAIN=0, SKIP_EVAL=0, USE_WANDB=1
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SEEDS_STR="${SEEDS:-0 1 2 3}"
SEEDS_STR="${SEEDS_STR//,/ }"
GPU_STR="${GPUS:-0,1,2,3}"
MODE="${MODE:-repro}"
INIT="${INIT:-scratch}"
RUN_ROOT="${RUN_ROOT:-results/runs/cosna/default}"
REPRO_ROOT="${REPRO_ROOT:-results/runs/bctd/tcsu20}"
BCTD_ROOT="${BCTD_ROOT:-results/runs/topostrand/bctd200}"
SEED0_SOURCE="${SEED0_SOURCE:-results/runs/topostrand/pilot200_bnd005_h2/seed_0/best_model_fixed3.pth}"
COSNA_INIT_ROOT="${COSNA_INIT_ROOT:-checkpoints/cosna_init}"
EPOCHS_ADD="${EPOCHS_ADD:-20}"
EPOCHS_TOTAL="${EPOCHS_TOTAL:-0}"
LOG_DIR="${LOG_DIR:-$REPO/results/logs}"
FORCE="${FORCE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"

abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$REPO/$1" ;;
  esac
}
RUN_ROOT_ABS=$(abspath "$RUN_ROOT")
REPRO_ROOT_ABS=$(abspath "$REPRO_ROOT")
BCTD_ROOT_ABS=$(abspath "$BCTD_ROOT")
SEED0_SOURCE_ABS=$(abspath "$SEED0_SOURCE")
COSNA_INIT_ROOT_ABS=$(abspath "$COSNA_INIT_ROOT")
if ! [[ "$EPOCHS_TOTAL" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EPOCHS_TOTAL must be a non-negative integer; got '$EPOCHS_TOTAL'" >&2
  exit 2
fi
if [ "$EPOCHS_TOTAL" -eq 0 ] && ! [[ "$EPOCHS_ADD" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EPOCHS_ADD must be positive when EPOCHS_TOTAL=0; got '$EPOCHS_ADD'" >&2
  exit 2
fi
case "$INIT" in
  scratch|cosna|bctd) ;;
  *) echo "ERROR: INIT must be scratch, cosna, or bctd; got '$INIT'" >&2; exit 2 ;;
esac

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
mkdir -p "$TMPDIR" "$LOG_DIR"

if [ "$USE_WANDB" != 0 ]; then
  source "$REPO/scripts/wandb_env.sh"
fi

read -r -a SEEDS <<< "$SEEDS_STR"
read -r -a GPUS <<< "${GPU_STR//,/ }"
if [ "${#SEEDS[@]}" -eq 0 ] || [ "${#GPUS[@]}" -eq 0 ]; then
  echo "ERROR: empty SEEDS/GPUS" >&2
  exit 2
fi
for seed in "${SEEDS[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEEDS must be non-negative integers; got '$seed'" >&2
    exit 2
  fi
done
for gpu in "${GPUS[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPUS must be non-negative integers; got '$gpu'" >&2
    exit 2
  fi
done
if [ "$(printf '%s\n' "${SEEDS[@]}" | sort -u | wc -l)" -ne "${#SEEDS[@]}" ]; then
  echo "ERROR: duplicate seed in SEEDS" >&2
  exit 2
fi
if [ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne "${#GPUS[@]}" ]; then
  echo "ERROR: duplicate GPU in GPUS" >&2
  exit 2
fi

run_one_seed() {
  local seed="$1" gpu="$2"
  local data="$REPO/data/sacm_3shot_seed${seed}"
  local work="$RUN_ROOT_ABS/seed_${seed}"
  local log="$LOG_DIR/cosna_baseline_seed${seed}.log"
  local source=""
  if [ "$INIT" = cosna ]; then
    source="$COSNA_INIT_ROOT_ABS/seed_${seed}/best_model.pth"
  elif [ "$INIT" = bctd ]; then
    if [ "$seed" = 0 ]; then
      source="$SEED0_SOURCE_ABS"
    else
      source="$BCTD_ROOT_ABS/seed_${seed}/best_model_fixed3.pth"
    fi
  fi
  mkdir -p "$work"
  : > "$log"
  echo "mode=$MODE seed=$seed gpu=$gpu work=$work" >> "$log"

  if [ "$MODE" = repro ]; then
    if [ ! -f "$work/best_model.pth" ] || [ "$FORCE" = 1 ]; then
      local repro_source="$REPRO_ROOT_ABS/seed_${seed}/best_model.pth"
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
      local init_args=()
      local source_args=()
      if [ "$INIT" = scratch ]; then
        init_args=(--init-mode scratch)
      else
        init_args=(--init-mode source)
        source_args=(--source-checkpoint "$source")
        if [ "$FORCE" = 1 ]; then
          init_args+=(--force-init)
        fi
      fi
      local epoch_args=()
      if [ "$EPOCHS_TOTAL" -gt 0 ]; then
        epoch_args=(--epochs-total "$EPOCHS_TOTAL")
      else
        epoch_args=(--epochs-add "$EPOCHS_ADD")
      fi
      local train_cmd=("$PYTHON" -m baseline_zoo.models.cosna.train
        --data-root "$data" --workdir "$work"
        "${source_args[@]}" "${epoch_args[@]}"
        "${init_args[@]}"
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
