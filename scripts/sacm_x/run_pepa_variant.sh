#!/usr/bin/env bash
# One-click runner for any full_pepa* preset.
#
# Usage:
#   bash scripts/sacm_x/run_pepa_variant.sh full_pepa_topo
#   bash scripts/sacm_x/run_pepa_variant.sh full_pepa_all_hr train
#   bash scripts/sacm_x/run_pepa_variant.sh full_pepa_hr smoke
#
# The first argument is the preset; the optional second argument is
# full | train | smoke (default: full).  Ranker presets automatically use
# the learned selector; all other presets keep fixed head 2.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

PRESET="${1:-full_pepa}"
MODE="${2:-full}"

# Never inherit an audit/eval output root from a diagnostic shell.  A stale
# OUTPUT_ROOT pointing at results/.../seed_0/eval makes the trainer write
# its checkpoints into the audit tree and can silently poison finalizers.
if [ -n "${OUTPUT_ROOT:-}" ]; then
  case "$OUTPUT_ROOT" in
    */eval|*selector_audit*|*audit*)
      echo "WARNING: ignoring inherited OUTPUT_ROOT=$OUTPUT_ROOT" >&2
      unset OUTPUT_ROOT
      ;;
  esac
fi
if [ -n "${CANONICAL_ROOT:-}" ]; then
  case "$CANONICAL_ROOT" in
    *selector_audit*|*audit*)
      echo "WARNING: ignoring inherited CANONICAL_ROOT=$CANONICAL_ROOT" >&2
      CANONICAL_ROOT=results/runs
      ;;
  esac
fi

GPU="${GPU:-0}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-300}"
USE_WANDB="${USE_WANDB:-true}"
INPUT_MODE="${INPUT_MODE:-resize}"
EVAL_INPUT_MODE="${EVAL_INPUT_MODE:-resize}"
USE_TTA="${USE_TTA:-false}"
VAL_FIXED_HEAD="${VAL_FIXED_HEAD:-2}"

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
    echo "ERROR: mode must be smoke | train | full (got '$MODE')" >&2
    exit 2
    ;;
esac

case "$PRESET" in
  *rank*|*all*|*v2*|*v3*)
    # Ranker/single-head presets use the IoU score directly; no fixed2.
    VAL_SELECTION=iou
    FIXED_HEAD=""
    SELECTION=iou
    ;;
  *)
    VAL_SELECTION=fixed
    VAL_FIXED_HEAD=2
    FIXED_HEAD=2
    SELECTION=""
    ;;
esac

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GPU SEED EPOCHS USE_WANDB SKIP_TEST CANONICAL_ROOT
export PRESET MONITOR=baseline
export VAL_SELECTION VAL_FIXED_HEAD FIXED_HEAD SELECTION
export EMPTY_GUARD=false VAL_EMPTY_GUARD=false
export INPUT_MODE EVAL_INPUT_MODE USE_TTA
# Preset-aware SCNP default: full_fixed2_scnp enables it unless the caller
# explicitly overrides SCNP=true/false.
if [ "${SCNP:-auto}" = "auto" ]; then
  case "$PRESET" in
    *scnp*) SCNP=true ;;
    *) SCNP=false ;;
  esac
fi
export SCNP

echo "============================================================"
echo " SACM-X PEPA variant: $PRESET"
echo " mode:       $MODE"
echo " gpu:        $GPU"
echo " seed:       $SEED"
echo " epochs:     $EPOCHS"
echo " geometry:   train=$INPUT_MODE eval=$EVAL_INPUT_MODE"
if [ -n "$FIXED_HEAD" ]; then
  echo " selection:  fixed head $FIXED_HEAD"
else
  echo " selection:  ranker/iou"
fi
echo " tta:        $USE_TTA"
echo " wandb:      $USE_WANDB"
echo " skip test:  $SKIP_TEST"
echo " repo:       $REPO"
echo "============================================================"

exec bash scripts/sacm_x/run_sacm_x_quick.sh
