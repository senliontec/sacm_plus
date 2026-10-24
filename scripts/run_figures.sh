#!/usr/bin/env bash
# Generate paper figures from results/paper/tables/*.csv.
#
# Usage:
#   bash scripts/run_figures.sh
#   bash scripts/run_figures.sh --tables-dir results/paper/tables \
#       --out-dir results/paper/figures --formats png,pdf
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
exec "$PYTHON" -m baseline_zoo.figures "$@"
