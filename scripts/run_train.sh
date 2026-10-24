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
exec "$PYTHON" -m baseline_zoo train "$@"
