#!/usr/bin/env bash
# Regenerate cross-model / cross-seed tables from existing metrics only.
#
# Usage:
#   bash scripts/run_report.sh --model all --seeds all
#   bash scripts/run_report.sh --model sacm_x --seeds 0
#   bash scripts/run_report.sh --model all --seeds all --dry-run
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
exec "$PYTHON" -m baseline_zoo report "$@"
