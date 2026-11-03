#!/usr/bin/env bash
# One-click inference-only OCTA native-resolution tiling evaluation.
#
# Does NOT train or modify checkpoints.  Tiles the original OCTA images,
# runs fixed head 2 of the existing SACM-X checkpoint, stitches probabilities,
# writes original-resolution predictions, then runs the standard prediction
# audit (Dice/IoU/clDice/HD95 + components).
#
# Fairness note: if these numbers are used in a paper table, apply the same
# tiling protocol to the compared baselines or report this as a separate
# inference-only setting.
#
# Usage:
#   GPU=0 bash scripts/sacm_x/run_octa_tiled_eval.sh
#
# Overrides: GPU, DATASET, CKPT, OUT, PRESET, TILE_STRIDE, BATCH, LIMIT, PYTHON
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
DATASET="${DATASET:-OCTA-500}"
PROTOCOL="${PROTOCOL:-data/sacm_3shot_seed0/protocol_manifest.csv}"
BASE="${BASE:-results/runs/sacm_x/full/seed_0}"
CKPT="${CKPT:-$BASE/best_model.pth}"
OUT="${OUT:-$REPO/results/octa_tiled_eval/seed_0}"
PRESET="${PRESET:-full}"
TILE_STRIDE="${TILE_STRIDE:-896}"
BATCH="${BATCH:-2}"
LIMIT="${LIMIT:-0}"

need_file() {
  [ -f "$1" ] || { echo "ERROR: missing $1" >&2; exit 2; }
}

need_file "$CKPT"
need_file "$PROTOCOL"
need_file "scripts/pipeline/protocol_specs.py"
need_file "scripts/sacm_x/eval_tiled_sacm_x.py"
need_file "scripts/pipeline/audit_predictions.py"

SPEC="$("$PYTHON" scripts/pipeline/protocol_specs.py \
  --protocol "$PROTOCOL" --datasets "$DATASET")"
if [ -z "$SPEC" ]; then
  echo "ERROR: protocol has no dataset $DATASET" >&2
  exit 2
fi
DATA_ROOT="${SPEC#*:}"

echo "repo:       $REPO"
echo "dataset:    $DATASET"
echo "data root:  $DATA_ROOT"
echo "checkpoint: $CKPT"
echo "preset:     $PRESET"
echo "tiling:     tile=1024 stride=$TILE_STRIDE batch=$BATCH"
echo "output:     $OUT"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON" scripts/sacm_x/eval_tiled_sacm_x.py \
  --checkpoint "$CKPT" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT/pred" \
  --dataset-name "$DATASET" \
  --preset "$PRESET" \
  --fixed-head 2 \
  --pred-threshold 0.0 \
  --tile-stride "$TILE_STRIDE" \
  --batch-size "$BATCH" \
  --limit "$LIMIT" \
  --device cuda

"$PYTHON" scripts/pipeline/audit_predictions.py \
  --pred-root "$OUT/pred" \
  --protocol "$PROTOCOL" \
  --out "$OUT/audit" \
  --datasets "$DATASET" \
  --workers 8

echo
echo "predictions: $OUT/pred/$DATASET"
echo "audit:       $OUT/audit/report.md"
echo "NOTE: apply the same tiling to compared methods before using this in the main table."
