#!/usr/bin/env bash
# One-click full_pepa run (PEPA TCSU + TADT on top of full SACM-X).
#
# Usage:
#   bash scripts/sacm_x/run_full_pepa.sh          # 800 epochs + 20-test eval
#   bash scripts/sacm_x/run_full_pepa.sh train    # 800 epochs, train only
#   bash scripts/sacm_x/run_full_pepa.sh smoke    # 1 epoch, no W&B, no eval
#
# Optional overrides:
#   GPU=0 SEED=0 EPOCHS=800 bash scripts/sacm_x/run_full_pepa.sh
#
# The wrapper keeps the same fixed-head/geometry policy as the current
# SACM-X comparison: fixed Stage-2 head 2, square resize, no empty guard.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

MODE="${1:-full}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-800}"
USE_WANDB="${USE_WANDB:-true}"

case "$MODE" in
  smoke)
    EPOCHS=1
    USE_WANDB=false
    SKIP_TEST=true
    CANONICAL_ROOT=results/runs_smoke
    ;;
  train)
    SKIP_TEST=true
    CANONICAL_ROOT="${CANONICAL_ROOT:-results/runs}"
    ;;
  full)
    SKIP_TEST=false
    CANONICAL_ROOT="${CANONICAL_ROOT:-results/runs}"
    ;;
  *)
    echo "ERROR: mode must be one of: smoke | train | full (got '$MODE')" >&2
    exit 2
    ;;
esac

export GPU SEED EPOCHS USE_WANDB SKIP_TEST CANONICAL_ROOT
export PRESET=full_pepa
export SCNP=false
export MONITOR=baseline

# Keep the comparison policy identical to the existing fixed2 runs.
export VAL_SELECTION=fixed
export VAL_FIXED_HEAD=2
export FIXED_HEAD=2
export EMPTY_GUARD=false
export VAL_EMPTY_GUARD=false
export INPUT_MODE=resize
export EVAL_INPUT_MODE=resize

echo "============================================================"
echo " SACM-X full_pepa (PEPA TCSU + TADT)"
echo " mode:       $MODE"
echo " gpu:        $GPU"
echo " seed:       $SEED"
echo " epochs:     $EPOCHS"
echo " geometry:   train=$INPUT_MODE eval=$EVAL_INPUT_MODE"
echo " selection:  fixed head 2"
echo " empty guard: false"
echo " wandb:      $USE_WANDB"
echo " skip test:  $SKIP_TEST"
echo " repo:       $REPO"
echo "============================================================"

exec bash scripts/sacm_x/run_sacm_x_quick.sh
