#!/usr/bin/env bash
# One-click: train the Deep Closing shape prior (if needed), then run the
# existing SACM-X train/eval quick path with Deep Closing post-processing.
#
# Usage:
#   GPU=0 bash scripts/sacm_x/run_sacm_x_deep_closing.sh
#   GPU=0 PRESET=full TRAIN_ROOT=data/sacm_3shot_seed1 \
#       OUTPUT_ROOT=results/sacm_x_dc_seed1 \
#       bash scripts/sacm_x/run_sacm_x_deep_closing.sh
#
# Environment:
#   GPU, TRAIN_ROOT, PRESET, OUTPUT_ROOT, EPOCHS, SEED,
#   DEEP_CLOSING_CKPT (default <OUTPUT_ROOT>/deep_closing/shape_prior.pth),
#   DEEP_EPOCHS (300), DEEP_RESOLUTION (512), DEEP_CHANNELS,
#   DEEP_CLOSING_THRESHOLD, RETRAIN_DEEP_CLOSING=true to force retraining.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

GPU="${GPU:-0}"
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot_seed0}"
PRESET="${PRESET:-full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/sacm_x_deep_closing}"
EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-42}"
DEEP_CLOSING_CKPT="${DEEP_CLOSING_CKPT:-$OUTPUT_ROOT/deep_closing/shape_prior.pth}"
DEEP_EPOCHS="${DEEP_EPOCHS:-300}"
DEEP_RESOLUTION="${DEEP_RESOLUTION:-512}"
DEEP_CHANNELS="${DEEP_CHANNELS:-16,32,64,128,256}"
DEEP_CLOSING_THRESHOLD="${DEEP_CLOSING_THRESHOLD:-}"
RETRAIN_DEEP_CLOSING="${RETRAIN_DEEP_CLOSING:-false}"

if [ ! -f "$TRAIN_ROOT/protocol_manifest.csv" ]; then
  echo "ERROR: missing protocol manifest: $TRAIN_ROOT/protocol_manifest.csv" >&2
  exit 2
fi

if [ "$RETRAIN_DEEP_CLOSING" = "true" ] || [ ! -f "$DEEP_CLOSING_CKPT" ]; then
  echo "== training Deep Closing shape prior =="
  GPU="$GPU" TRAIN_ROOT="$TRAIN_ROOT" OUTPUT="$DEEP_CLOSING_CKPT" \
    EPOCHS="$DEEP_EPOCHS" RESOLUTION="$DEEP_RESOLUTION" \
    CHANNELS="$DEEP_CHANNELS" \
    bash scripts/sacm_x/train_deep_closing.sh
else
  echo "== using existing Deep Closing checkpoint: $DEEP_CLOSING_CKPT =="
fi
if [ ! -f "$DEEP_CLOSING_CKPT" ]; then
  echo "ERROR: Deep Closing checkpoint was not created: $DEEP_CLOSING_CKPT" >&2
  exit 2
fi

echo "== SACM-X train/eval with Deep Closing post-processing =="
GPU="$GPU" TRAIN_ROOT="$TRAIN_ROOT" PRESET="$PRESET" \
  OUTPUT_ROOT="$OUTPUT_ROOT" EPOCHS="$EPOCHS" SEED="$SEED" \
  DEEP_CLOSING_CKPT="$DEEP_CLOSING_CKPT" \
  DEEP_CLOSING_THRESHOLD="$DEEP_CLOSING_THRESHOLD" \
  bash scripts/sacm_x/run_sacm_x_quick.sh
