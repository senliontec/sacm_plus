#!/usr/bin/env bash
# Sequential 20-test evaluation (one ViT-L in RAM at a time).
# Parallel eval previously exhausted host RAM/swap and was killed.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/dev/anaconda3/envs/tsai/bin/python}"
SEED="${SEED:-0}"
ROOT="${NIGHT_ROOT:-results/runs/night_final}"
LOG_DIR="${LOG_DIR:-results/ablation_logs}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval_final}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run_one() {
  local gpu="$1" preset="$2" log="$3"
  local train_out="$ROOT/sacm_x/$preset/seed_$SEED"
  echo "===== $preset on GPU $gpu -> $train_out/$EVAL_SUBDIR"
  PYTHON="$PY" GPU="$gpu" SEED="$SEED" PRESET="$preset" \
  TRAIN_ROOT="data/sacm_3shot_seed${SEED}" \
  TRAIN_OUT="$train_out" OUTPUT_ROOT="$train_out/$EVAL_SUBDIR" \
  SAM_CKPT="checkpoints/sam_vit_l_0b3195.pth" \
  MATCH_BASELINE_GEOMETRY=1 EMPTY_GUARD=false \
  REUSE_EVAL_MODEL=1 PAPER_METRICS_ONLY=1 NUM_WORKERS=0 \
  USE_WANDB=false SELECTION=iou FIXED_HEAD= \
    bash scripts/sacm_x/eval_sacm_x_20.sh > "$log" 2>&1
}

LOG_V3="$LOG_DIR/night_eval_seq_v3_${STAMP}.log"
LOG_HY="$LOG_DIR/night_eval_seq_hy_${STAMP}.log"
echo "v3 log:     $LOG_V3"
echo "hybrid log: $LOG_HY"
run_one 0 full_v3 "$LOG_V3"
run_one 1 full_v2_skel_aux "$LOG_HY"
echo
echo "Tables:"
echo "  $ROOT/sacm_x/full_v3/seed_$SEED/$EVAL_SUBDIR/tables"
echo "  $ROOT/sacm_x/full_v2_skel_aux/seed_$SEED/$EVAL_SUBDIR/tables"
