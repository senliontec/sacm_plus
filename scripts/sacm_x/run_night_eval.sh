#!/usr/bin/env bash
# Evaluate the two night-final checkpoints on all 20 test sets.
# Matched geometry (resize, no empty guard) for the SACM+ comparison.
#
# Usage:
#   bash scripts/sacm_x/run_night_eval.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

PY="${PYTHON:-/home/dev/anaconda3/envs/tsai/bin/python}"
SEED="${SEED:-0}"
ROOT="${NIGHT_ROOT:-results/runs/night_final}"
LOG_DIR="${LOG_DIR:-results/ablation_logs}"
GPU_V3="${GPU_V3:-0}"
GPU_HYBRID="${GPU_HYBRID:-1}"
# W&B + DataLoader fork is unsafe (ForkedError in workers); eval does not
# need W&B.  workers=0 also avoids the fork entirely.
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval_retry}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_V3="$LOG_DIR/night_eval_v3_${STAMP}.log"
LOG_HY="$LOG_DIR/night_eval_hybrid_${STAMP}.log"

run_one() {
  local gpu="$1" preset="$2" log="$3"
  local train_out="$ROOT/sacm_x/$preset/seed_$SEED"
  PYTHON="$PY" GPU="$gpu" SEED="$SEED" PRESET="$preset" \
  TRAIN_ROOT="data/sacm_3shot_seed${SEED}" \
  TRAIN_OUT="$train_out" OUTPUT_ROOT="$train_out/$EVAL_SUBDIR" \
  SAM_CKPT="checkpoints/sam_vit_l_0b3195.pth" \
  MATCH_BASELINE_GEOMETRY=1 EMPTY_GUARD=false \
  REUSE_EVAL_MODEL=1 PAPER_METRICS_ONLY=1 NUM_WORKERS="$NUM_WORKERS" \
  USE_WANDB=false SELECTION=iou FIXED_HEAD= \
    bash scripts/sacm_x/eval_sacm_x_20.sh > "$log" 2>&1
}

echo "============================================================"
echo " night eval: full_v3 (GPU $GPU_V3) / full_v2_skel_aux (GPU $GPU_HYBRID)"
echo " root: $ROOT"
echo " logs: $LOG_V3"
echo "       $LOG_HY"
echo "============================================================"

run_one "$GPU_V3" full_v3 "$LOG_V3" &
PID_V3=$!
run_one "$GPU_HYBRID" full_v2_skel_aux "$LOG_HY" &
PID_HY=$!

set +e
wait "$PID_V3"; RC_V3=$?
wait "$PID_HY"; RC_HY=$?
set -e
echo "v3 eval exit=$RC_V3  log=$LOG_V3"
echo "hybrid eval exit=$RC_HY log=$LOG_HY"
if [ "$RC_V3" -ne 0 ] || [ "$RC_HY" -ne 0 ]; then
  echo "===== v3 eval tail ====="; tail -n 60 "$LOG_V3" || true
  echo "===== hybrid eval tail ====="; tail -n 60 "$LOG_HY" || true
  exit 1
fi
echo
echo "Tables:"
echo "  $ROOT/sacm_x/full_v3/seed_$SEED/$EVAL_SUBDIR/tables"
echo "  $ROOT/sacm_x/full_v2_skel_aux/seed_$SEED/$EVAL_SUBDIR/tables"
