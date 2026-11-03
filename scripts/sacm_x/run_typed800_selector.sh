#!/usr/bin/env bash
# One-click typed-candidate run:
#   full_typed_topo (typed heads + Betti loss)
#   + fixed head 2 for validation/checkpoint selection
#   + fixed head 2 for the one-shot 20-dataset evaluation
#
# This is the first recommended experiment after the audit.  It deliberately
# does NOT tune anything on the test sets: the validation policy is fixed,
# and the test evaluation is one-shot.
#
# Usage:
#   GPU=7 SEED=0 bash scripts/sacm_x/run_typed800_selector.sh
#   GPU=7 SEED=0 EPOCHS=800 SKIP_TEST=false \
#     bash scripts/sacm_x/run_typed800_selector.sh
#   GPU=7 SEED=0 VAL_SELECTION=quality \
#     bash scripts/sacm_x/run_typed800_selector.sh
#
# Defaults:
#   PRESET=full_typed_topo
#   VAL_SELECTION=fixed
#   VAL_FIXED_HEAD=3   (typed h3 "recall"; h0 is the best-clDice head)
#   VAL_EMPTY_GUARD=true
#   EPOCHS=800
#
# Set VAL_SELECTION=iou/gating/quality to compare selectors under the same
# typed+topology training recipe.  Set TOPOLOGY_LOSS=none to disable the
# auxiliary topology term while keeping the typed candidates.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

GPU="${GPU:-0}"
SEED="${SEED:-0}"
PRESET="${PRESET:-full_typed_topo}"
VAL_SELECTION="${VAL_SELECTION:-fixed}"
VAL_FIXED_HEAD="${VAL_FIXED_HEAD:-3}"
VAL_EMPTY_GUARD="${VAL_EMPTY_GUARD:-true}"
EPOCHS="${EPOCHS:-800}"
SKIP_TEST="${SKIP_TEST:-false}"
TOPOLOGY_LOSS="${TOPOLOGY_LOSS:-}"
TOPOLOGY_LOSS_WEIGHT="${TOPOLOGY_LOSS_WEIGHT:-0.05}"
TOPOLOGY_RESOLUTION="${TOPOLOGY_RESOLUTION:-256}"

export GPU SEED PRESET VAL_SELECTION VAL_FIXED_HEAD VAL_EMPTY_GUARD
export EPOCHS SKIP_TEST TOPOLOGY_LOSS TOPOLOGY_LOSS_WEIGHT TOPOLOGY_RESOLUTION

echo "typed-candidate run: preset=$PRESET seed=$SEED gpu=$GPU"
echo "validation policy: selection=$VAL_SELECTION fixed_head=$VAL_FIXED_HEAD empty_guard=$VAL_EMPTY_GUARD"
echo "topology override: ${TOPOLOGY_LOSS:-<preset default>} weight=$TOPOLOGY_LOSS_WEIGHT resolution=$TOPOLOGY_RESOLUTION"
echo "epochs=$EPOCHS skip_test=$SKIP_TEST"

exec bash scripts/sacm_x/run_sacm_x_quick.sh
