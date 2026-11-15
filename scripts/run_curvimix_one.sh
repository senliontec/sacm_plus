#!/usr/bin/env bash
# One seed of the independent CurviMix-SAM model.
#
# Usage:
#   SEED=0 GPU=0 EPOCHS=200 EVAL=1 bash scripts/run_curvimix_one.sh
#
# Env:
#   SEED (required), GPU=0, EVAL_GPU=GPU, EPOCHS=200, DATA_ROOT
#   RUN_ROOT=results/runs/curvimix/epochs${EPOCHS}
#   COPY_PASTE_PROB=0.5, USE_WANDB=0
#   EVAL=1, TTA=0, TTA_SELECT=0, TAG
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SEED="${SEED:?set SEED}"
GPU="${GPU:-0}"
EVAL_GPU="${EVAL_GPU:-$GPU}"
EPOCHS="${EPOCHS:-200}"
DATA_ROOT="${DATA_ROOT:-$REPO/data/sacm_3shot_seed${SEED}}"
RUN_ROOT="${RUN_ROOT:-$REPO/results/runs/curvimix/epochs${EPOCHS}}"
WORKDIR="$RUN_ROOT/seed_${SEED}"
COPY_PASTE_PROB="${COPY_PASTE_PROB:-0.5}"
USE_WANDB="${USE_WANDB:-0}"
EVAL="${EVAL:-1}"
TTA="${TTA:-0}"
TTA_SELECT="${TTA_SELECT:-0}"
TAG="${TAG:-}"

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$REPO/results/logs"

if [ "$USE_WANDB" != 0 ]; then
  source "$REPO/scripts/wandb_env.sh"
fi

mkdir -p "$WORKDIR"
LOG="$REPO/results/logs/curvimix_seed${SEED}_${EPOCHS}.log"
echo "CurviMix-SAM seed=$SEED epochs=$EPOCHS gpu=$GPU workdir=$WORKDIR" > "$LOG"

"$PYTHON" -m baseline_zoo.run_one --model curvimix --stage train \
  --data-root "$DATA_ROOT" --workdir "$WORKDIR" \
  --epochs "$EPOCHS" --seed "$SEED" --device "cuda:$GPU" \
  --use_wandb "$USE_WANDB" \
  --sam-checkpoint "$REPO/checkpoints/sam_vit_l_0b3195.pth" \
  --batch-size 1 --accum-steps 1 --lr 0.0003 --weight-decay 0.01 \
  --grad-clip 1.0 --scheduler official --selection dice \
  --bce-weight 0.7 --dice-weight 0.3 --bce-pos-weight 2 \
  --binary-targets 0 --deploy-threshold 0.5 \
  --deep-sup-weight 0.3 --specialized-heads 1 \
  --specialized-head-weight 0.1 --specialized-topology-weight 0.1 \
  --head-fusion 1 --fusion-loss-weight 1.0 \
  --stage2-loss-scope all --topology-head all \
  --cl-dice-weight 0.1 --skeleton-recall-weight 0.1 \
  --topology-resolution 512 --topology-warmup-epochs 5 \
  --head-select gating --deployment-head 3 --adapter-dim-ratio 0.1 \
  --num-workers 0 --save-every 0 --train-aug mild \
  --copy-paste-prob "$COPY_PASTE_PROB" \
  --wandb_project sacm --wandb_entity buaazqk \
  --wandb_group "curvimix_${EPOCHS}" --wandb_name "curvimix-seed${SEED}" \
  >> "$LOG" 2>&1

"$PYTHON" -m baseline_zoo.models.curvimix.derive_fixed3 \
  --input "$WORKDIR/best_model.pth" \
  --output "$WORKDIR/best_model_fixed3.pth" \
  --deployment-head 3 >> "$LOG" 2>&1

if [ "$EVAL" != 0 ]; then
  eval_args=("$PYTHON" -m baseline_zoo.models.curvimix.evaluate
    --data-root "$DATA_ROOT"
    --protocol "$DATA_ROOT/protocol_manifest.csv"
    --workdir "$WORKDIR"
    --checkpoint "$WORKDIR/best_model_fixed3.pth"
    --seed "$SEED" --device "cuda:$EVAL_GPU"
    --use-wandb "$USE_WANDB")
  if [ "$TTA" != 0 ]; then eval_args+=(--tta-flip); fi
  if [ "$TTA_SELECT" != 0 ]; then eval_args+=(--tta-select); fi
  if [ -n "$TAG" ]; then eval_args+=(--tag "$TAG"); fi
  "${eval_args[@]}" >> "$LOG" 2>&1
fi

echo "DONE seed=$SEED workdir=$WORKDIR log=$LOG"
