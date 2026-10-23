#!/usr/bin/env bash
# One-click SAM ViT-L + U-Net: train -> 20-dataset prediction -> metrics ->
# visualization -> print paper metrics.
#
# Usage:
#   bash scripts/run_sam_vit_l_unet_all.sh                 # train 200 epochs + evaluate
#   SKIP_TRAIN=1 bash scripts/run_sam_vit_l_unet_all.sh    # evaluate existing checkpoint
#   SKIP_TRAIN=1 SKIP_PREDICT=1 bash scripts/run_sam_vit_l_unet_all.sh  # metrics only
#   DATASETS="DRIVE CHASEDB1 WIRE" bash scripts/run_sam_vit_l_unet_all.sh
#   GPU=1 EPOCHS=200 bash scripts/run_sam_vit_l_unet_all.sh
#   USE_WANDB=false bash scripts/run_sam_vit_l_unet_all.sh
#
# Environment:
#   PYTHON, GPU, EPOCHS, BATCH_SIZE, USE_WANDB, SKIP_TRAIN, SKIP_PREDICT,
#   DATASETS, CHECKPOINT, SAM_CKPT
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

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
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
USE_WANDB="${USE_WANDB:-false}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_PREDICT="${SKIP_PREDICT:-0}"
DATASETS="${DATASETS:-}"
CHECKPOINT="${CHECKPOINT:-workdirs/baselines/sam_vit_l_unet/best_model.pth}"
SAM_CKPT="${SAM_CKPT:-checkpoints/sam_vit_l_0b3195.pth}"

if [ "$SKIP_TRAIN" != "1" ]; then
  echo "== [1/3] train =="
  BACKGROUND=false GPU="$GPU" EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" \
    USE_WANDB="$USE_WANDB" KILL_GPU_PROCS=false KILL_OLD=false \
    PYTHON="$PYTHON" bash scripts/run_sam_vit_l_unet_train.sh
else
  echo "== [1/3] train skipped (SKIP_TRAIN=1) =="
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

echo "== [2/3] predict + paper metrics + visualizations =="
runner_args=(scripts/run_baseline_test.py --method sam_vit_l_unet
  --checkpoint "$CHECKPOINT" --sam-checkpoint "$SAM_CKPT" --no-resume)
if [ "$SKIP_PREDICT" = "1" ]; then
  runner_args+=(--skip-predict)
fi
if [ -n "$DATASETS" ]; then
  read -r -a dataset_array <<< "$DATASETS"
  runner_args+=(--datasets "${dataset_array[@]}")
fi
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${runner_args[@]}"

echo "== [3/3] paper metrics =="
"$PYTHON" scripts/show_paper_metrics.py --method sam_vit_l_unet

echo ""
echo "outputs:"
echo "  predictions: results/predictions/sam_vit_l_unet/"
echo "  metrics:     results/paper/paper_metrics.csv"
echo "  macro:       results/paper/paper_macro.csv"
echo "  tables:      results/paper/paper_tables.md"
echo "  vis:         results/visualizations/sam_vit_l_unet/"
