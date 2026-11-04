#!/usr/bin/env bash
# Launch ONLY the final SACM-X pipeline on physical GPUs 0-4.
#
# This wrapper exists so the final run can be launched independently from
# the SACM official run (GPU 5-7).  It calls the same component pipeline as
# scripts/run_all_tonight.sh, but never starts the official baseline.
#
# Usage:
#   bash scripts/run_final_gpu0_4.sh
#
# Optional overrides:
#   OUT_ROOT=results/runs/my_final bash scripts/run_final_gpu0_4.sh
#   EPOCHS=800 TRAIN_SEEDS="0 1 2 3 4" bash scripts/run_final_gpu0_4.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

require_file() {
  if [ ! -e "$1" ]; then
    echo "ERROR: missing $1" >&2
    exit 2
  fi
}
require_file scripts/sacm_x/run_component_full.sh
require_file scripts/sacm_x/component_repair_eval.py
require_file src/models/sacm_x/configs.py
require_file src/train/trainer.py
require_file checkpoints/sam_vit_l_0b3195.pth

# Fail before spending GPU time if the server copy is older than the code
# this pipeline depends on.
if ! grep -q 'full_fixed2_gate' src/models/sacm_x/configs.py; then
  echo "ERROR: server code is missing the full_fixed2_gate preset." >&2
  echo "Upload the current src/models/sacm_x/configs.py first." >&2
  exit 2
fi
if ! grep -q -- '--no_augment' src/train/trainer.py; then
  echo "ERROR: server trainer has no --no_augment flag." >&2
  echo "Upload the current src/train/trainer.py first." >&2
  exit 2
fi
if ! grep -q 'freeze_experts' src/models/sam_network/residual_moe.py; then
  echo "ERROR: server residual MoE has no gate-only expert freeze." >&2
  echo "Upload the current src/models/sam_network/residual_moe.py and src/train/trainer.py." >&2
  exit 2
fi

# A stale CUDA mask must not redirect the per-seed jobs.  The component
# pipeline itself sets CUDA_VISIBLE_DEVICES=0..4 for its five children.
unset CUDA_VISIBLE_DEVICES || true
unset OUTPUT_ROOT || true

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export OUT_ROOT="${OUT_ROOT:-results/runs/final_gpu0_4_${STAMP}}"
export PRESET="${PRESET:-full_fixed2_gate}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3 4}"
export PER_GPU_DATA_SEED="${PER_GPU_DATA_SEED:-1}"
export EPOCHS="${EPOCHS:-800}"
export VAL_INTERVAL="${VAL_INTERVAL:-1}"
export NO_AUGMENT=1
export RUN_NO_GATE="${RUN_NO_GATE:-0}"
export USE_WANDB="${USE_WANDB:-true}"

echo "============================================================"
echo " FINAL SACM-X on physical GPUs 0-4"
echo " preset:    $PRESET"
echo " data:      seed <-> GPU 0-4 (PER_GPU_DATA_SEED=$PER_GPU_DATA_SEED)"
echo " seeds:     $TRAIN_SEEDS"
echo " epochs:    $EPOCHS"
echo " augment:   disabled"
echo " output:    $OUT_ROOT"
echo " logs:      $OUT_ROOT/logs/"
echo "============================================================"

exec bash scripts/sacm_x/run_component_full.sh
