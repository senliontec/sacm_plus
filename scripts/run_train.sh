#!/usr/bin/env bash
# Train specified models/seeds through the unified registry.
#
# Usage:
#   bash scripts/run_train.sh --model sacm_x --seeds 0
#   bash scripts/run_train.sh --model unet,sacm_x --seeds 0 1 --gpus 2,3
#   bash scripts/run_train.sh --model all --seeds all --gpus 0,1,2,3 --resume
#   bash scripts/run_train.sh --model unet --seeds 0 --dry-run
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

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Default W&B: load the self-hosted env and require credentials unless the
# caller explicitly chooses offline/disabled.  The configs already pass
# --use_wandb for every baseline.
source "$PWD/scripts/sacm_x/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done
if [[ "$DRY_RUN" -eq 0 && "$WANDB_MODE" != "offline" && "$WANDB_MODE" != "disabled" ]]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled by default but no credentials were found." >&2
    echo "Set WANDB_API_KEY or run 'wandb login', or use WANDB_MODE=offline/disabled." >&2
    exit 1
  fi
fi

exec "$PYTHON" -m baseline_zoo train "$@"
