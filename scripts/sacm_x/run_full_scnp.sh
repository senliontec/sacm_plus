#!/usr/bin/env bash
# One-click full_scnp run. No environment variables are required.
#
# Usage:
#   bash scripts/sacm_x/run_full_scnp.sh          # 300 epochs + 20-test eval
#   bash scripts/sacm_x/run_full_scnp.sh train    # 300 epochs, train only
#   bash scripts/sacm_x/run_full_scnp.sh smoke    # 1 epoch, no W&B, no eval
#
# Optional overrides (defaults are the comparison protocol):
#   GPU=0 SEED=0 EPOCHS=300 bash scripts/sacm_x/run_full_scnp.sh
#
# This wrapper fixes the configuration that produced the current best SACM-X
# result: fixed Stage-2 head 2, square resize for train and eval, no empty
# guard, baseline per-epoch monitoring, and the full_scnp preset.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

MODE="${1:-full}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-300}"
USE_WANDB="${USE_WANDB:-true}"

case "$MODE" in
  smoke)
    # Isolated output root: a 1-epoch smoke must never touch the canonical
    # 300-epoch checkpoint under results/runs/.
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
export PRESET=full_scnp
export SCNP=true
export MONITOR=baseline

# Selection policy: fixed head 2 for both checkpoint selection and final eval.
export VAL_SELECTION=fixed
export VAL_FIXED_HEAD=2
export FIXED_HEAD=2
export EMPTY_GUARD=false
export VAL_EMPTY_GUARD=false

# Matched geometry: square resize everywhere, exactly as in the current
# fixed2 + matched-geometry comparison.
export INPUT_MODE=resize
export EVAL_INPUT_MODE=resize

echo "============================================================"
echo " SACM-X full_scnp"
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
