#!/usr/bin/env bash
# Launch ONLY SACM official on physical GPUs 5-7.
#
# This wrapper exists so the official baseline can be started independently
# from the final SACM-X run (GPU 0-4).  It trains seeds 0-2 from scratch and
# then evaluates all 20 test datasets automatically.
#
# Usage:
#   bash scripts/run_official_gpu5_7.sh
#
# Optional overrides:
#   SEEDS="0 1 2" GPU_LIST=5,6,7 bash scripts/run_official_gpu5_7.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

require_file() {
  if [ ! -e "$1" ]; then
    echo "ERROR: missing $1" >&2
    exit 2
  fi
}
require_file scripts/tools/run_sacm_official_5to7.sh
require_file configs/models/sacm_official.yaml
require_file checkpoints/sam_vit_l_0b3195.pth

# The official command uses physical device ids (cuda:5/cuda:6/cuda:7).
# Clear an inherited visibility mask so those ids are not remapped.
unset CUDA_VISIBLE_DEVICES || true

export GPU_LIST="${GPU_LIST:-5,6,7}"
export SEEDS="${SEEDS:-0 1 2}"
export MODEL="${MODEL:-sacm_official}"

echo "============================================================"
echo " SACM official on physical GPUs 5-7"
echo " model:   $MODEL"
echo " seeds:   $SEEDS"
echo " epochs:  800 (configs/models/sacm_official.yaml)"
echo " augment: disabled (official recipe)"
echo " output:  results/runs/sacm_official/default/seed_<seed>/"
echo "============================================================"

exec bash scripts/tools/run_sacm_official_5to7.sh
