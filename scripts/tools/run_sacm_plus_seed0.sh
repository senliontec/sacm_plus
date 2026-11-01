#!/usr/bin/env bash
# One-click SACM+ (fixed official SACM) train + full 20-dataset eval.
#
# Usage:
#   GPU=0 bash scripts/tools/run_sacm_plus_seed0.sh
#   GPU=0 SEED=1 EPOCHS=800 bash scripts/tools/run_sacm_plus_seed0.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-800}"
DATA_ROOT="${DATA_ROOT:-data/sacm_3shot_seed${SEED}}"
WORKDIR="${WORKDIR:-results/workdirs/baselines/sacm_plus/seed_${SEED}}"
PRED_ROOT="${PRED_ROOT:-results/predictions/sacm_plus/seed_${SEED}}"
PAPER_ROOT="${PAPER_ROOT:-results/paper/sacm_plus/seed_${SEED}}"
PROTOCOL="${PROTOCOL:-${DATA_ROOT}/protocol_manifest.csv}"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$DATA_ROOT/split_manifest.csv" ]]; then
  echo "missing split manifest: $DATA_ROOT/split_manifest.csv" >&2
  exit 2
fi
if [[ ! -f "$PROTOCOL" ]]; then
  echo "missing protocol manifest: $PROTOCOL" >&2
  exit 2
fi

python -m baseline_zoo.run_one \
  --model sacm_plus --stage train \
  --data-root "$DATA_ROOT" --workdir "$WORKDIR" \
  --epochs "$EPOCHS" --seed "$SEED" --device "cuda:${GPU}" --use_wandb

python scripts/pipeline/run_baseline_test.py \
  --method sacm_plus \
  --protocol "$PROTOCOL" \
  --seed "$SEED" \
  --checkpoint "$WORKDIR/best_model.pth" \
  --pred-root "$PRED_ROOT" \
  --paper-root "$PAPER_ROOT" \
  --device "cuda:${GPU}" --use_wandb true --skip-visualize
