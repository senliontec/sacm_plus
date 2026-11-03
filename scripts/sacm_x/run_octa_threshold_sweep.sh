#!/usr/bin/env bash
# OCTA threshold calibration sweep (eval-only).
#
# Runs fixed head 2 with matched geometry (resize, no empty guard) on
# OCTA-500, records quick metrics at several logit thresholds, and summarizes
# Dice/recall/components.  This does NOT retrain and does NOT modify the
# model.  The sweep is diagnostic; choose a threshold on validation, never
# on the test set.
#
# Usage:
#   bash scripts/sacm_x/run_octa_threshold_sweep.sh
#
# Environment overrides:
#   GPU, SEED, DATASET, OUT, THRESHOLDS, PYTHON
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
DATASET="${DATASET:-OCTA-500}"
PRESET="${PRESET:-full}"
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot_seed0}"
BASE="${BASE:-$REPO/results/runs/sacm_x/$PRESET/seed_$SEED}"
OUT="${OUT:-$REPO/results/octa_threshold_sweep/seed_$SEED}"
THRESHOLDS="${THRESHOLDS:--1.0 -0.5 -0.25 0.0 0.25 0.5 1.0}"

CKPT="${CKPT:-$BASE/best_model.pth}"

need_file() {
  [ -f "$1" ] || { echo "ERROR: missing $1" >&2; exit 2; }
}

need_file "$CKPT"
need_file "$TRAIN_ROOT/protocol_manifest.csv"
need_file "scripts/sacm_x/eval_sacm_x_20.sh"
need_file "scripts/sacm_x/summarize_threshold_sweep.py"

echo "repo:       $REPO"
echo "checkpoint: $CKPT"
echo "dataset:    $DATASET"
echo "geometry:   resize (matched to plus), no empty guard"
echo "policy:     fixed head 2"
echo "thresholds: $THRESHOLDS"
echo "output:     $OUT"

rc=0
PYTHON="$PYTHON" \
GPU="$GPU" \
SEED="$SEED" \
PRESET="$PRESET" \
TRAIN_OUT="$BASE" \
CKPT="$CKPT" \
TRAIN_ROOT="$TRAIN_ROOT" \
OUTPUT_ROOT="$OUT/eval" \
CANONICAL_ROOT="$OUT/canonical" \
FIXED_HEAD=2 \
SELECTION= \
EMPTY_GUARD=false \
EVAL_INPUT_MODE=resize \
USE_WANDB=false \
DATASETS="$DATASET" \
REUSE_EVAL_MODEL=1 \
PAPER_METRICS_ONLY=1 \
COMPONENT_METRICS=true \
AUDIT_THRESHOLDS="$THRESHOLDS" \
bash scripts/sacm_x/eval_sacm_x_20.sh || rc=$?

CSV="$(find "$OUT" -name threshold_audit.csv -print -quit 2>/dev/null || true)"
if [ -z "$CSV" ]; then
  echo "ERROR: no threshold_audit.csv under $OUT" >&2
  if [ "$rc" -ne 0 ]; then
    exit "$rc"
  fi
  exit 1
fi

"$PYTHON" scripts/sacm_x/summarize_threshold_sweep.py \
  --input "$CSV" \
  --out-dir "$OUT/analysis" \
  --baseline 0.0

echo
echo "threshold audit: $CSV"
echo "summary:         $OUT/analysis/threshold_summary.md"
echo "NOTE: do not choose a test-set threshold for the paper;"
echo "      calibrate the final threshold on validation/support data."
if [ "$rc" -ne 0 ]; then
  echo "WARNING: eval pipeline exited $rc, but the threshold audit was found." >&2
  exit "$rc"
fi
