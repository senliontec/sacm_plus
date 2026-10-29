#!/usr/bin/env bash
# One-click overnight CurvField screening (seeds 0/1/2, 100 epochs, full 20-dataset test).
# Usage: bash scripts/curvfield/run_night.sh [--dry-run]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PATH="/home/dev/anaconda3/envs/tsai/bin:$PATH"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec python scripts/curvfield/run_night.py "$@"
