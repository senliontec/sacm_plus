#!/usr/bin/env bash
# Build the SACM-Table-2-style per-dataset paper table from recorded runs.
#
# Run after both one-click reproduction scripts have finished:
#   bash scripts/run_dcd_seeds03_gpu0123.sh
#   bash scripts/run_sacm_official_seeds03_gpu4567.sh
#   bash scripts/build_paper_table.sh
#   bash scripts/build_paper_table.sh --no-percent --decimals 4
#
# Outputs (default results/reports/paper_table2/):
#   table2.tex           means only, percent, best bold / second underlined
#   table2_with_std.tex  mean ± std LaTeX
#   table2.md            Markdown, mean ± std
#   table2_means.md      Markdown, means only, percent
#   table2.csv           long per-dataset table (raw fractions + unit)
#   table2_macro.csv     Seen/Unseen/Novel/All macro rows
#
# Env overrides:
#   MODELS="dcd:all_dense sacm_official:default"
#   SEEDS="0 1 2 3"
#   DATASETS="CHASEDB1 CrackTree200 ... WIRE"   optional analysis subset
#   OUT_DIR=results/reports/paper_table2
#   EXPECTED_DATASETS=20
#   PYTHON=...
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

MODELS="${MODELS:-dcd:all_dense sacm_official:default}"
SEEDS="${SEEDS:-0 1 2 3}"
OUT_DIR="${OUT_DIR:-results/reports/paper_table2}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"
DATASETS="${DATASETS:-}"
read -r -a MODEL_ARGS <<< "$MODELS"
read -r -a SEED_ARGS <<< "$SEEDS"
DATASET_ARGS=()
DATASET_FLAGS=()
if [ -n "$DATASETS" ]; then
  read -r -a DATASET_ARGS <<< "$DATASETS"
  DATASET_FLAGS=(--datasets "${DATASET_ARGS[@]}")
fi

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  elif [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
export PYTHON
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

echo "============================================================"
echo " Paper Table 2 builder"
echo " models: $MODELS"
echo " seeds:  $SEEDS"
echo " output: $OUT_DIR"
echo "============================================================"

"$PYTHON" scripts/pipeline/build_paper_table.py \
  --repo-root "$REPO" \
  --models "${MODEL_ARGS[@]}" \
  --seeds "${SEED_ARGS[@]}" \
  --expected-datasets "$EXPECTED_DATASETS" \
  --out-dir "$OUT_DIR" \
  "${DATASET_FLAGS[@]}" \
  "$@"

echo "============================================================"
echo " paper table: $OUT_DIR/table2.tex"
echo " with std:    $OUT_DIR/table2_with_std.tex"
echo " preview:     $OUT_DIR/table2_means.md"
echo "============================================================"
