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

# Match every baseline entrypoint: use the shared self-hosted W&B server.
source "$REPO_ROOT/scripts/sacm_x/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
if [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B mode=$WANDB_MODE but no credentials were found." >&2
    echo "Set WANDB_API_KEY or run 'wandb login', or use WANDB_MODE=offline/disabled." >&2
    exit 1
  fi
  if command -v curl >/dev/null 2>&1; then
    _wandb_http="$(curl -s -o /dev/null -m 8 -w '%{http_code}' \
      "$WANDB_BASE_URL" || true)"
    case "$_wandb_http" in
      2*|3*) ;;
      *)
        echo "ERROR: W&B server $WANDB_BASE_URL is unreachable (HTTP ${_wandb_http:-none})." >&2
        exit 1
        ;;
    esac
  fi
fi

exec python scripts/curvfield/run_night.py "$@"
