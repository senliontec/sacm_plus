#!/usr/bin/env bash
# One-click SACM+ (fixed official SACM) on GPU5-7: seeds 0-2.
#
# This is intentionally separate from the final SACM-X evaluation on GPU0-4
# and from sacm_official, which is already complete.  It uses the standard
# baseline wrappers, so W&B auth, 800-epoch config, and automatic 20-dataset
# evaluation use the canonical SACM+ run layout.
#
# Usage (server):
#   bash scripts/tools/run_sacm_plus_5to7.sh
#
# If SACM+ checkpoints already exist and you only want evaluation, use:
#   bash scripts/run_eval.sh --model sacm_plus --seeds 0 1 2 --gpus 5,6,7
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU_LIST="${GPU_LIST:-5,6,7}"
SEEDS="${SEEDS:-0 1 2}"
MODEL="${MODEL:-sacm_plus}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a SEED_ARGS <<< "$SEEDS"

if [ "$DRY_RUN" = "1" ]; then
  bash scripts/run_train.sh --model "$MODEL" --seeds "${SEED_ARGS[@]}" \
    --gpus "$GPU_LIST" --dry-run
  bash scripts/run_eval.sh --model "$MODEL" --seeds "${SEED_ARGS[@]}" \
    --gpus "$GPU_LIST" --dry-run
  exit 0
fi

echo "============================================================"
echo " SACM+ train + 20-dataset test"
echo " model:  $MODEL"
echo " gpus:   $GPU_LIST"
echo " seeds:  $SEEDS"
echo " repo:   $REPO_ROOT"
echo "============================================================"

# Training is from scratch (no --resume).  The runtime backs up an existing
# best_model.pth before starting a new run.
bash scripts/run_train.sh --model "$MODEL" --seeds "${SEED_ARGS[@]}" \
  --gpus "$GPU_LIST"

bash scripts/run_eval.sh --model "$MODEL" --seeds "${SEED_ARGS[@]}" \
  --gpus "$GPU_LIST"

echo "SACM+ train+test done."
echo "Checkpoints: results/runs/sacm_plus/default/seed_{0,1,2}/best_model.pth"
echo "Eval tables: results/runs/sacm_plus/default/seed_{0,1,2}/eval/tables/"
