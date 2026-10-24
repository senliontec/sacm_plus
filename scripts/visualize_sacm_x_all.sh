#!/usr/bin/env bash
# Export all SACM-X test predictions as binary masks + overlays.
#
# Usage:
#   bash scripts/visualize_sacm_x_all.sh
#   DATASETS="CHCrack5k WIRE" bash scripts/visualize_sacm_x_all.sh
#
# Output:
#   results/visualizations/sacm_x_all/<dataset>/masks/*.png
#   results/visualizations/sacm_x_all/<dataset>/overlay/*.png
#   results/visualizations/sacm_x_all/<dataset>/manifest.csv
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-results/sacm_x_official_200}"
PRESET="${PRESET:-full}"
VIZ_ROOT="${VIZ_ROOT:-results/visualizations/sacm_x_all}"
PROTOCOL="${PROTOCOL:-data/sacm_3shot/protocol_manifest.csv}"
DATASETS="${DATASETS:-}"

ARGS=(
  --eval-root "$OUTPUT_ROOT"
  --preset "$PRESET"
  --protocol "$PROTOCOL"
  --out-root "$VIZ_ROOT"
)
if [ -n "$DATASETS" ]; then
  read -r -a dataset_array <<< "$DATASETS"
  ARGS+=(--datasets "${dataset_array[@]}")
fi

"$PYTHON" scripts/visualize_sacm_x_all.py "${ARGS[@]}"
echo "done: $VIZ_ROOT"
