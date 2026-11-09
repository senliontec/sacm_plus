#!/usr/bin/env bash
# One-click entrypoint for the config-driven SACM-STC ablation.
#
# The experiment matrix (arms, epochs, flags, contrasts, diagnostics) lives
# in configs/ablations/sacm_stc_scope_structure.yaml.  This wrapper only
# selects the interpreter, loads the self-hosted W&B environment, and calls
# the generic Python runner.
#
# Usage (server, repo root):
#   bash scripts/run_sacm_stc_ablation.sh --stage audit --gpus 0,1,2,3
#   bash scripts/run_sacm_stc_ablation.sh --stage all --gpus 0,1,2,3,4,5,6,7
#   bash scripts/run_sacm_stc_ablation.sh --stage eval --gpus 0,1 --arms b3_all_dense_all_structure
#   bash scripts/run_sacm_stc_ablation.sh --stage all --dry-run
#
# Stages: audit | train | eval | diagnose | all | analyze.  The default is
# the safe candidate audit; training starts only when --stage train/all is
# explicit.  --resume is true by default; pass --no-resume to rerun completed
# stages.
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

# All stages that can launch training/evaluation use the same self-hosted
# W&B setup as run_train.sh/run_eval.sh.  Audit/diagnose/analyze are
# inference-only and do not require credentials.
STAGE="audit"
DRY_RUN=0
prev=""
for arg in "$@"; do
  if [ "$prev" = "--stage" ]; then STAGE="$arg"; fi
  if [[ "$arg" == --stage=* ]]; then STAGE="${arg#--stage=}"; fi
  if [ "$arg" = "--dry-run" ]; then DRY_RUN=1; fi
  prev="$arg"
done
case "$STAGE" in
  train|eval|all)
    source "$REPO/scripts/wandb_env.sh"
    export WANDB_MODE="${WANDB_MODE:-online}"
    if [ "$DRY_RUN" -eq 0 ] \
       && [ "$WANDB_MODE" != "offline" ] \
       && [ "$WANDB_MODE" != "disabled" ]; then
      if ! wandb_auth_ok; then
        echo "ERROR: W&B is enabled by default but no credentials were found." >&2
        echo "Set WANDB_API_KEY or run 'wandb login', or use WANDB_MODE=offline/disabled." >&2
        exit 1
      fi
    fi
    ;;
esac

exec "$PYTHON" scripts/pipeline/run_sacm_stc_ablation.py "$@"
