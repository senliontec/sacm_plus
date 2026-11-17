#!/usr/bin/env bash
# One-click StrandCal-SAM 200-epoch train + eval.
#
# Usage:
#   bash scripts/run_strandcal_seeds03_200.sh
#   SKIP_TRAIN=1 bash scripts/run_strandcal_seeds03_200.sh
#   DRY_RUN=1 bash scripts/run_strandcal_seeds03_200.sh
#
# Env:
#   SEEDS="0 1 2 3", TRAIN_GPUS=0,1,2,3, EVAL_GPUS=4,5,6,7
#   SKIP_TRAIN=0, SKIP_EVAL=0, DRY_RUN=0
#   WANDB_MODE=online (default), WANDB_PROJECT=sacm, WANDB_ENTITY=buaazqk
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

SEEDS="${SEEDS:-0 1 2 3}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
EVAL_GPUS="${EVAL_GPUS:-4,5,6,7}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a SEED_ARGS <<< "${SEEDS//,/ }"
read -r -a TRAIN_ARGS <<< "${TRAIN_GPUS//,/ }"
read -r -a EVAL_ARGS <<< "${EVAL_GPUS//,/ }"
DRY_ARGS=()
if [ "$DRY_RUN" = 1 ]; then
  DRY_ARGS=(--dry-run)
fi

if [ "$SKIP_TRAIN" != 1 ]; then
  bash scripts/run_train.sh --model strandcal \
    --seeds "${SEED_ARGS[@]}" --gpus "$(IFS=,; echo "${TRAIN_ARGS[*]}")" \
    "${DRY_ARGS[@]}"
fi

if [ "$SKIP_EVAL" != 1 ]; then
  bash scripts/run_eval.sh --model strandcal \
    --seeds "${SEED_ARGS[@]}" --gpus "$(IFS=,; echo "${EVAL_ARGS[*]}")" \
    "${DRY_ARGS[@]}"
fi

echo "DONE StrandCal-SAM seeds=${SEEDS}"
