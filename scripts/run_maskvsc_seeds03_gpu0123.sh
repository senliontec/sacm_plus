#!/usr/bin/env bash
# One-click MaskVSC 3-shot baseline: seeds 0-3 on four GPUs, then 20-dataset eval.
#
# Uses the deployed task-adapted 600+200 recipe from configs/models/maskvsc.yaml:
# InstanceNorm, batch 1, mild augmentation, BCE+Connect_Loss+soft Dice (1.0)
# for stage 1; clean-input BCE + zero-initialised residual ViG for stage 2.
#
# Usage (server, repo root):
#   bash scripts/run_maskvsc_seeds03_gpu0123.sh
#   GPUS=4,5,6,7 bash scripts/run_maskvsc_seeds03_gpu0123.sh
#   SEEDS="0 1" GPUS=0,1 bash scripts/run_maskvsc_seeds03_gpu0123.sh
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

GPUS="${GPUS:-0,1,2,3}"
SEEDS="${SEEDS:-0 1 2 3}"

echo "[maskvsc] stage 1: 600 epochs, stage 2: 200 epochs"
echo "[maskvsc] training seeds: $SEEDS on GPUs $GPUS"
# shellcheck disable=SC2086
bash scripts/run_train.sh --model maskvsc --seeds $SEEDS --gpus "$GPUS"

echo "[maskvsc] evaluating seeds: $SEEDS on GPUs $GPUS"
# shellcheck disable=SC2086
bash scripts/run_eval.sh --model maskvsc --seeds $SEEDS --gpus "$GPUS"

echo "[maskvsc] done."
echo "[maskvsc] build the paper table with: bash scripts/build_paper_table.sh"
