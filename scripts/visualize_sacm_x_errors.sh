#!/usr/bin/env bash
# Visualize the worst SACM-X predictions on original images.
#
# Usage:
#   bash scripts/visualize_sacm_x_errors.sh
#   DATASET=WIRE TOP_N=12 bash scripts/visualize_sacm_x_errors.sh
#
# Output:
#   results/visualizations/sacm_x_errors/<dataset>/
#     worst_<N>_contact.png
#     01_<image>_viz.png ...
#     worst_manifest.csv
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

DATASET="${DATASET:-CHCrack5k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/sacm_x_official_200}"
PRESET="${PRESET:-full}"
TOP_N="${TOP_N:-8}"
VIZ_ROOT="${VIZ_ROOT:-results/visualizations/sacm_x_errors}"
PROTOCOL="${PROTOCOL:-data/sacm_3shot/protocol_manifest.csv}"
METRICS="$OUTPUT_ROOT/$PRESET/$DATASET/metrics.csv"

if [ ! -f "$METRICS" ]; then
  echo "ERROR: missing $METRICS" >&2
  echo "Wait until $DATASET evaluation finishes, or check DATASET/OUTPUT_ROOT." >&2
  exit 2
fi

OUT_DIR="$VIZ_ROOT/$DATASET"
"$PYTHON" scripts/visualize_sacm_x_errors.py \
  --dataset "$DATASET" \
  --eval-root "$OUTPUT_ROOT" \
  --preset "$PRESET" \
  --protocol "$PROTOCOL" \
  --out-dir "$OUT_DIR" \
  --top-n "$TOP_N"

echo
echo "done: $OUT_DIR"
