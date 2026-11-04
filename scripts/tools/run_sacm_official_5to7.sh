#!/usr/bin/env bash
# One-click SACM official: train seeds 0-2 on GPU 5-7, then test all 20
# datasets automatically.
#
# Usage:
#   bash scripts/tools/run_sacm_official_5to7.sh
#   GPU_LIST=5,6,7 SEEDS="0 1 2" bash scripts/tools/run_sacm_official_5to7.sh
#   DRY_RUN=1 bash scripts/tools/run_sacm_official_5to7.sh
#
# Training uses the existing baseline wrappers, which source the self-hosted
# W&B environment.  Official SACM defaults to 800 epochs from its model YAML.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU_LIST="${GPU_LIST:-5,6,7}"
SEEDS="${SEEDS:-0 1 2}"
MODEL="${MODEL:-sacm_official}"
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
echo " SACM official: train + 20-dataset test"
echo " model:  $MODEL"
echo " gpus:   $GPU_LIST"
echo " seeds:  $SEEDS"
echo " repo:   $REPO_ROOT"
echo "============================================================"

# No --resume: retrain from scratch.  The runtime backs up an existing
# best_model.pth before starting a new run.
bash scripts/run_train.sh --model "$MODEL" --seeds "${SEED_ARGS[@]}" \
  --gpus "$GPU_LIST"

# The baseline eval entrypoint derives all 20 test roots from the protocol
# manifest and writes predictions/tables under the canonical run roots.
bash scripts/run_eval.sh --model "$MODEL" --seeds "${SEED_ARGS[@]}" \
  --gpus "$GPU_LIST"

echo "SACM official train+test done."
echo "Checkpoints: results/runs/sacm_official/seed_{0,1,2}/best_model.pth"
echo "Eval tables: results/runs/sacm_official/seed_{0,1,2}/eval/tables/"
