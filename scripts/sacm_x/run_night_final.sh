#!/usr/bin/env bash
# Overnight final comparison: full_v3 vs full_v2_skel_aux.
# Both start from the base SAM ViT-L checkpoint and train 800 epochs.
#
#   GPU 0: full_v3
#   GPU 1: full_v2_skel_aux
#
# Usage: bash scripts/sacm_x/run_night_final.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

PY="${PYTHON:-/home/dev/anaconda3/envs/tsai/bin/python}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-800}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
GPU_V3="${GPU_V3:-0}"
GPU_HYBRID="${GPU_HYBRID:-1}"
ROOT="${NIGHT_ROOT:-results/runs/night_final}"
LOG_DIR="${LOG_DIR:-results/ablation_logs}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot_seed${SEED}}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
mkdir -p "$LOG_DIR" "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_V3="$LOG_DIR/night_v3_${STAMP}.log"
LOG_HY="$LOG_DIR/night_hybrid_${STAMP}.log"

echo "============================================================"
echo " night final: full_v3 vs full_v2_skel_aux"
echo " gpus:       $GPU_V3 (v3) / $GPU_HYBRID (hybrid)"
echo " epochs:     $EPOCHS  val every $VAL_INTERVAL"
echo " seed:       $SEED"
echo " ckpt:       $CKPT"
echo " root:       $ROOT"
echo " logs:       $LOG_V3"
echo "             $LOG_HY"
echo "============================================================"

GPU="$GPU_V3" SEED="$SEED" EPOCHS="$EPOCHS" VAL_INTERVAL="$VAL_INTERVAL" \
CKPT="$CKPT" TRAIN_ROOT="$TRAIN_ROOT" CANONICAL_ROOT="$ROOT" OUTPUT_ROOT= \
USE_WANDB="$USE_WANDB" WANDB_MODE="$WANDB_MODE" NUM_WORKERS="${NUM_WORKERS:-2}" \
WANDB_NAME="sacm_x_local_seed${SEED}_full_v3_800" \
  bash scripts/sacm_x/run_pepa_variant.sh full_v3 train \
  > "$LOG_V3" 2>&1 &
PID_V3=$!

GPU="$GPU_HYBRID" SEED="$SEED" EPOCHS="$EPOCHS" VAL_INTERVAL="$VAL_INTERVAL" \
CKPT="$CKPT" TRAIN_ROOT="$TRAIN_ROOT" CANONICAL_ROOT="$ROOT" OUTPUT_ROOT= \
USE_WANDB="$USE_WANDB" WANDB_MODE="$WANDB_MODE" NUM_WORKERS="${NUM_WORKERS:-2}" \
WANDB_NAME="sacm_x_local_seed${SEED}_full_v2_skel_aux_800" \
  bash scripts/sacm_x/run_full_v2_skel_aux.sh train \
  > "$LOG_HY" 2>&1 &
PID_HY=$!

set +e
wait "$PID_V3"; RC_V3=$?
wait "$PID_HY"; RC_HY=$?
set -e

echo "v3 exit=$RC_V3  log=$LOG_V3"
echo "hybrid exit=$RC_HY  log=$LOG_HY"
if [ "$RC_V3" -ne 0 ] || [ "$RC_HY" -ne 0 ]; then
  echo "===== v3 tail ====="; tail -n 60 "$LOG_V3" || true
  echo "===== hybrid tail ====="; tail -n 60 "$LOG_HY" || true
  exit 1
fi

V3_HIST="$ROOT/sacm_x/full_v3/seed_${SEED}/metrics_history.json"
HY_HIST="$ROOT/sacm_x/full_v2_skel_aux/seed_${SEED}/metrics_history.json"
if [ -f "$V3_HIST" ] && [ -f "$HY_HIST" ]; then
  "$PY" scripts/sacm_x/compare_val_runs.py \
    "full_v3=$V3_HIST" "full_v2_skel_aux=$HY_HIST" || true
fi

if [ "${NIGHT_EVAL:-1}" = "1" ]; then
  echo "== night 20-dataset evaluation =="
  NIGHT_ROOT="$ROOT" SEED="$SEED" PYTHON="$PY" \
    bash scripts/sacm_x/run_night_eval.sh || {
      echo "night eval failed" >&2
      exit 1
    }
fi
