#!/usr/bin/env bash
# One-click DCD 800-epoch main run + component ablation.
#
# What it runs
# ------------
#   4 arms x 4 seeds x 800 epochs = 16 training jobs on 8 GPUs
#     ssh              canonical DCD (specialised heads + fusion)
#     nofusion         no fusion training
#     native_control   native four-head control
#     notopo           no topology supervision
#   then derives the inference-only fixed candidate 3 for every arm
#   then evaluates on the 20 protocol test datasets:
#     - every arm at fixed candidate 3
#     - DCD gating+fusion and native gating (same checkpoints, eval only)
#   then writes results/ablations/dcd_full_800/summary.{md,csv,html,json}
#   and, if the official SACM tables exist, the standard
#   fixed3-vs-SACM report under results/ablations/dcd_full_800/fixed3_vs_sacm.
#
# Usage (server, repo root, preferably in tmux):
#   bash scripts/run_dcd_full_800.sh
#   DRY_RUN=1 bash scripts/run_dcd_full_800.sh --dry-run
#   SKIP_TRAIN=1 bash scripts/run_dcd_full_800.sh       # eval/resume only
#   RESUME=0 bash scripts/run_dcd_full_800.sh           # retrain from scratch
#
# Environment overrides:
#   SEEDS="0 1 2 3"       EPOCHS=800
#   TRAIN_GPUS=0,1,2,3,4,5,6,7
#   EVAL_GPUS=0,1,2,3,4,5,6,7
#   CONFIG=configs/ablations/dcd_full_800.yaml
#   OFFICIAL_ROOT=results/runs/sacm/default
#   SUMMARY_ROOT=results/ablations/dcd_full_800
#   RESUME=1              FORCE_DERIVE=0  FORCE_EVAL=0
#   SKIP_TRAIN=0          SKIP_EVAL=0     SKIP_STANDARD_SUMMARY=0
#   REQUIRE_OFFICIAL=1    EXPECTED_DATASETS=20
#   MAX_USED_MIB=2000     MIN_FREE_GB=80  ALLOW_BUSY=0
#   ALLOW_LOW_DISK=0      EVAL_WORKERS=<run_baseline_test default>
#   WANDB_MODE=online     WANDB_PROJECT=sacm  WANDB_ENTITY=buaazqk
#   PYTHON=...            LOG_DIR=...
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

for arg in "$@"; do
  if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
    exec "$PYTHON" scripts/pipeline/run_dcd_full_800.py "$@"
  fi
done

if [ "$DRY_RUN" = 1 ]; then
  exec "$PYTHON" scripts/pipeline/run_dcd_full_800.py "$@"
fi

if [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled by default but no credentials were found." >&2
    echo "Set WANDB_API_KEY or run 'wandb login', or use " \
         "WANDB_MODE=offline/disabled." >&2
    exit 1
  fi
fi

LOCK_DIR="$REPO/results/runs/.locks"
mkdir -p "$LOCK_DIR"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_DIR/dcd_full_800.lock"
  if ! flock -n 9; then
    echo "ERROR: another DCD full 800 driver is already running" >&2
    exit 3
  fi
else
  LOCK_MK="$LOCK_DIR/dcd_full_800.lockdir"
  if ! mkdir "$LOCK_MK" 2>/dev/null; then
    echo "ERROR: another DCD full 800 driver is already running" >&2
    exit 3
  fi
  trap 'rmdir "$LOCK_MK" 2>/dev/null || true' EXIT
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$REPO/results/logs/dcd_full_800_${STAMP}}"
export LOG_DIR
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1

echo "============================================================"
echo " DCD full 800 | 4 arms x 4 seeds | 20-dataset evaluation"
echo " repo:       ${REPO}"
echo " python:     ${PYTHON}"
echo " W&B mode:   ${WANDB_MODE}"
echo " driver log: ${LOG_DIR}/driver.log"
echo "============================================================"

exec "$PYTHON" scripts/pipeline/run_dcd_full_800.py "$@"
