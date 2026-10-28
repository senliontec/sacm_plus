#!/usr/bin/env bash
# One-click read-only A2SAM branch/DCA1 triage.
#
# Usage:
#   GPU=0 bash scripts/tools/run_a2sam_triage.sh
#   GPU=1 CHECKPOINT=... DATA_ROOT=... OUTPUT=... \
#       bash scripts/tools/run_a2sam_triage.sh
#
# Extra arguments are passed to a2sam_triage.py, e.g.:
#   ... run_a2sam_triage.sh --a2sam-datasets all --splits val
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

GPU_SPEC="${GPU:-0}"
CHECKPOINT="${CHECKPOINT:-results/paper/a2sam/seed_0/best_model.pth}"
DATA_ROOT="${DATA_ROOT:-data/sacm_3shot_seed0}"
OUTPUT="${OUTPUT:-results/diagnostics/a2sam_seed_0}"
PYTHON="${PYTHON:-python}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: missing checkpoint: $CHECKPOINT" >&2
  echo "Set CHECKPOINT=/path/to/best_model.pth" >&2
  exit 2
fi
if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: missing data root: $DATA_ROOT" >&2
  echo "Set DATA_ROOT=/path/to/data/sacm_3shot_seed0" >&2
  exit 2
fi

echo "[a2sam-triage] gpu=$GPU_SPEC checkpoint=$CHECKPOINT"
echo "[a2sam-triage] data_root=$DATA_ROOT output=$OUTPUT"
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES="$GPU_SPEC" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
"$PYTHON" scripts/tools/a2sam_triage.py \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT" \
  --device cuda:0 \
  --sweep \
  "$@"
