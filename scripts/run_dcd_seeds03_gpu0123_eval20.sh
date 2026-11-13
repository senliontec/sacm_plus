#!/usr/bin/env bash
# One-click DCD run:
#   seeds 0 1 2 3, GPUs 0 1 2 3, 200 epochs
#   train -> derive fixed candidate-3 -> evaluate all 20 datasets -> summary
#
# Usage (server, repo root, preferably under tmux):
#   bash scripts/run_dcd_seeds03_gpu0123_eval20.sh
#   DRY_RUN=1 bash scripts/run_dcd_seeds03_gpu0123_eval20.sh
#   bash scripts/run_dcd_seeds03_gpu0123_eval20.sh --dry-run
#
# Any environment override accepted by run_dcd_expert_200.sh still works,
# e.g. EPOCHS=800, RESUME=0, WANDB_MODE=offline.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

export SEEDS="${SEEDS:-0 1 2 3}"
export TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
export EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
export EPOCHS="${EPOCHS:-200}"
export EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"

for arg in "$@"; do
  case "$arg" in
    --dry-run) export DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

exec bash scripts/run_dcd_expert_200.sh
