#!/usr/bin/env bash
# One-click lean fixed3 ablation:
#   train nofusion / native_control / notopo (4 seeds, 800 epochs)
#   derive fixed H3 for nofusion + notopo
#   evaluate fixed3 / gating+fusion / nofusion / native_control / notopo
#   write results/ablations/sacm_gcd_fixed3_lean/summary.{csv,md,html}
#
# Usage (server, repo root):
#   bash scripts/run_sacm_gcd_fixed3_lean.sh
#   bash scripts/run_sacm_gcd_fixed3_lean.sh --dry-run
#   SKIP_TRAIN=1 bash scripts/run_sacm_gcd_fixed3_lean.sh
#   TRAIN_GPUS=0,1,2,3 EVAL_GPUS=4,5,6,7 bash scripts/run_sacm_gcd_fixed3_lean.sh
#
# The Python driver reads configs/ablations/sacm_gcd_fixed3_lean.yaml and
# uses the existing training/eval pipeline.  It does not modify model code.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  elif [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
export PYTHON
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-$REPO/results/.tmp}"
mkdir -p "$TMPDIR"

source "$REPO/scripts/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
export WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"

DRY_RUN="${DRY_RUN:-0}"
for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    DRY_RUN=1
  fi
done
case "${DRY_RUN,,}" in
  1|true|yes|on) DRY_RUN=1 ;;
  0|false|no|off) DRY_RUN=0 ;;
  *) echo "ERROR: DRY_RUN must be 0/1" >&2; exit 2 ;;
esac
if [ "$DRY_RUN" = 0 ] \
   && [ "$WANDB_MODE" != "offline" ] \
   && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled by default but no credentials were found." >&2
    echo "Set WANDB_API_KEY or run 'wandb login', or use WANDB_MODE=offline/disabled." >&2
    exit 1
  fi
fi

exec "$PYTHON" scripts/pipeline/run_fixed3_lean_ablation.py "$@"
