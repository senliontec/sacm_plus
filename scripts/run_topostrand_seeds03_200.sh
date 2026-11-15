#!/usr/bin/env bash
# One-click TopoStrand-SAM 200-epoch main run.
#
# It trains seeds 0-3, derives fixed3 checkpoints, evaluates the 20 datasets
# with label-free resolution routing + 4-view flip TTA, and writes per-seed
# tables under results/runs/topostrand/<variant>/seed_<s>/eval_routed.
#
# Usage:
#   bash scripts/run_topostrand_seeds03_200.sh
#   DRY_RUN=1 bash scripts/run_topostrand_seeds03_200.sh
#
# Env:
#   SEEDS="0 1 2 3", EPOCHS=200
#   TRAIN_GPUS=0,1,2,3, EVAL_GPUS=4,5,6,7
#   RUN_ROOT=results/runs/topostrand/final200_pa
#   USE_WANDB=1, WANDB_PROJECT=sacm, WANDB_ENTITY=buaazqk
#   SKIP_TRAIN=0, SKIP_EVAL=0, DRY_RUN=0, RESUME=0
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SEEDS_STR="${SEEDS:-0 1 2 3}"
SEEDS_STR="${SEEDS_STR//,/ }"
EPOCHS="${EPOCHS:-200}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
EVAL_GPUS="${EVAL_GPUS:-4,5,6,7}"
RUN_ROOT="${RUN_ROOT:-results/runs/topostrand/final200_pa}"
WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"
WANDB_GROUP="${WANDB_GROUP:-topostrand_${EPOCHS}}"
USE_WANDB="${USE_WANDB:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-$REPO/results/.tmp}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "$TMPDIR" "$REPO/results/logs"

if [ "$USE_WANDB" != 0 ]; then
  source "$REPO/scripts/wandb_env.sh"
fi

read -r -a SEEDS <<< "$SEEDS_STR"
read -r -a TGPUS <<< "${TRAIN_GPUS//,/ }"
read -r -a EGPUS <<< "${EVAL_GPUS//,/ }"
if [ "${#SEEDS[@]}" -eq 0 ] || [ "${#TGPUS[@]}" -eq 0 ] || [ "${#EGPUS[@]}" -eq 0 ]; then
  echo "ERROR: empty SEEDS/TRAIN_GPUS/EVAL_GPUS" >&2
  exit 2
fi

run_train() {
  local seed="$1" gpu="$2" workdir="$3"
  local cmd=("$PYTHON" -m methods.topostrand.run_one --stage train
    --data-root "$REPO/data/sacm_3shot_seed${seed}" --workdir "$workdir"
    --epochs "$EPOCHS" --seed "$seed" --device "cuda:${gpu}"
    --use_wandb "$USE_WANDB"
    --sam-checkpoint "$REPO/checkpoints/sam_vit_l_0b3195.pth"
    --bce-weight 0.7 --dice-weight 0.3 --bce-pos-weight 2
    --binary-targets 0 --deploy-threshold 0.5
    --stage2-loss-scope all --topology-head all
    --cl-dice-weight 0.1 --skeleton-recall-weight 0.1
    --specialized-heads 1 --specialized-head-weight 0.1
    --specialized-topology-weight 0.1
    --head-fusion 1 --fusion-loss-weight 1.0
    --topology-resolution 512 --topology-warmup-epochs 5
    --deep-sup-weight 0.3 --head-select gating --deployment-head 3
    --adapter-dim-ratio 0.1 --num-workers 0 --save-every 0
    --train-aug mild --copy-paste-prob 0.5
    --deployed-tversky-weight 0.1 --area-loss-weight 0.1
    --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY"
    --wandb_group "$WANDB_GROUP" --wandb_name "topostrand-seed${seed}")
  if [ "$RESUME" = 1 ]; then
    cmd+=(--resume)
  fi
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY train seed=%s gpu=%s: %q ' "$seed" "$gpu" "${cmd[0]}"
    printf '%q ' "${cmd[@]:1}"; printf '\n'
    return 0
  fi
  mkdir -p "$workdir"
  local log="$REPO/results/logs/topostrand_seed${seed}_${EPOCHS}.log"
  echo "train seed=$seed gpu=$gpu workdir=$workdir" > "$log"
  "${cmd[@]}" >> "$log" 2>&1
}

run_eval() {
  local seed="$1" gpu="$2" workdir="$3"
  local cmd=("$PYTHON" -m methods.topostrand.evaluate_routed
    --data-root "$REPO/data/sacm_3shot_seed${seed}"
    --protocol "$REPO/data/sacm_3shot_seed${seed}/protocol_manifest.csv"
    --workdir "$workdir"
    --checkpoint "$workdir/best_model_fixed3.pth"
    --seed "$seed" --device "cuda:${gpu}" --use-wandb 0
    --tta-flip --tile 1024 --stride 896
    --routing-skel-ratio 1.0 --routing-area-ratio 0.7
    --tag _routed)
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY eval seed=%s gpu=%s: %q ' "$seed" "$gpu" "${cmd[0]}"
    printf '%q ' "${cmd[@]:1}"; printf '\n'
    return 0
  fi
  "${cmd[@]}" >> "$REPO/results/logs/topostrand_seed${seed}_${EPOCHS}.log" 2>&1
}

workdir_for() {
  local seed="$1"
  case "$RUN_ROOT" in
    /*) printf '%s/seed_%s\n' "$RUN_ROOT" "$seed" ;;
    *) printf '%s/%s/seed_%s\n' "$REPO" "$RUN_ROOT" "$seed" ;;
  esac
}

pids=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${TGPUS[$((i % ${#TGPUS[@]}))]}"
  workdir="$(workdir_for "$seed")"
  if [ "$SKIP_TRAIN" != 1 ]; then
    run_train "$seed" "$gpu" "$workdir" &
    pids+=("$!")
  fi
done
for pid in "${pids[@]:-}"; do wait "$pid"; done

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  workdir="$(workdir_for "$seed")"
  if [ "$SKIP_TRAIN" != 1 ]; then
    if [ "$DRY_RUN" = 1 ]; then
      echo "DRY derive seed=$seed"
    else
      "$PYTHON" -m methods.topostrand.derive_fixed3 \
        --input "$workdir/best_model.pth" \
        --output "$workdir/best_model_fixed3.pth" \
        --deployment-head 3 >> "$REPO/results/logs/topostrand_seed${seed}_${EPOCHS}.log" 2>&1
    fi
  fi
done

if [ "$SKIP_EVAL" != 1 ]; then
  pids=()
  for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    gpu="${EGPUS[$((i % ${#EGPUS[@]}))]}"
    workdir="$(workdir_for "$seed")"
    run_eval "$seed" "$gpu" "$workdir" &
    pids+=("$!")
  done
  for pid in "${pids[@]:-}"; do wait "$pid"; done
fi

echo "DONE TopoStrand seeds=${SEEDS[*]} run_root=$RUN_ROOT"
