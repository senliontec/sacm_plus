#!/usr/bin/env bash
# Evaluate specified models/seeds through the unified registry.
# This never trains; it requires the checkpoint declared by each model config.
#
# Usage:
#   bash scripts/run_eval.sh --model sacm_x --seeds 0
#   bash scripts/run_eval.sh --model unet,sacm_official --seeds 0 1
#   bash scripts/run_eval.sh --model all --seeds all --gpus 0,1,2,3 --resume
#   bash scripts/run_eval.sh --model unet --seeds 0 --dry-run
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
exec "$PYTHON" -m baseline_zoo eval "$@"
