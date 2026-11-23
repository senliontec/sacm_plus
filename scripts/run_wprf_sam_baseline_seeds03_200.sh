#!/usr/bin/env bash
# One-click WPRF-SAM baseline (3-shot, 20 datasets, 200 epochs, V3 recipe).
#
# Usage:
#   bash scripts/run_wprf_sam_baseline_seeds03_200.sh
#   SEEDS="0 1" TRAIN_GPUS=0,1 EVAL_GPUS=0,1 \
#     bash scripts/run_wprf_sam_baseline_seeds03_200.sh
#   SKIP_TRAIN=1 bash scripts/run_wprf_sam_baseline_seeds03_200.sh
#
# Env: SEEDS, TRAIN_GPUS, EVAL_GPUS, SKIP_TRAIN, SKIP_EVAL, DRY_RUN.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

SEEDS="${SEEDS:-0 1 2 3}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EXTRA=()
if [ "${DRY_RUN:-0}" = "1" ]; then
  EXTRA+=(--dry-run)
fi

COMMON=(--model wprf_sam --seeds ${SEEDS})

if [ "$SKIP_TRAIN" != "1" ]; then
  bash scripts/run_train.sh "${COMMON[@]}" --gpus "$TRAIN_GPUS" "${EXTRA[@]}"
fi
if [ "$SKIP_EVAL" != "1" ]; then
  bash scripts/run_eval.sh "${COMMON[@]}" --gpus "$EVAL_GPUS" "${EXTRA[@]}"
fi
