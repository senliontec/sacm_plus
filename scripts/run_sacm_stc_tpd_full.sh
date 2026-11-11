#!/usr/bin/env bash
# One-click SACM-STC full + TPD effectiveness run (800 epochs, 4 seeds).
#
# Control (same 800-epoch full recipe, no TPD), 20-dataset test, 4 seeds:
#   sacm_stc full : Dice 56.30  IoU 42.69  clDice 62.78
#   sacm_official : Dice 55.40  IoU 41.85  clDice 60.34
#
# The new model is the full recipe plus TPD (image dense prompt,
# topology-quality ranking, quality-vetted refinement).
#
# Usage (repo root):
#   bash scripts/run_sacm_stc_tpd_full.sh              # 4 GPUs, seeds 0-3
#   GPU_LIST=0,1 SEEDS="0 1" bash scripts/run_sacm_stc_tpd_full.sh
#   bash scripts/run_sacm_stc_tpd_full.sh --dry-run
#
# Env overrides: GPU_LIST, SEEDS, ARM, MAX_USED_MIB, ALLOW_BUSY,
#                TPD_FULL_LOG, TPD_FULL_DAEMON (internal)
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

CONFIG="configs/ablations/sacm_stc_tpd_full.yaml"
read -r -a SEED_ARR <<< "${SEEDS:-0 1 2 3}"
ARM="${ARM:-tpd_full}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
LOG_ROOT="$REPO/results/logs/sacm_stc_tpd_full"
LOCK_FILE="$LOG_ROOT/run.lock"
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ ! -f "$REPO/$CONFIG" ]; then
  echo "ERROR: missing config: $REPO/$CONFIG" >&2
  exit 2
fi
if ! grep -q '^epochs: 800$' "$REPO/$CONFIG"; then
  echo "ERROR: full-effectiveness config must set epochs: 800" >&2
  exit 2
fi
if ! grep -q '^eval_checkpoint: best_model.pth$' "$REPO/$CONFIG"; then
  echo "ERROR: full-effectiveness config must use best_model.pth" >&2
  exit 2
fi
if [ ! -f "$REPO/checkpoints/sam_vit_l_0b3195.pth" ]; then
  echo "ERROR: missing SAM ViT-L checkpoint:" \
       "$REPO/checkpoints/sam_vit_l_0b3195.pth" >&2
  exit 2
fi
for flag in --image-dense-prompt --topology-quality \
            --quality-refine-min-score-ratio; do
  if ! grep -q -- "$flag" "$REPO/src/baseline_zoo/adapters/sacm_stc.py"; then
    echo "ERROR: adapter is missing $flag; sync the adapter and tpd.py." >&2
    exit 2
  fi
done
if [ "${#SEED_ARR[@]}" -eq 0 ]; then
  echo "ERROR: SEEDS is empty" >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
if [ "${#GPU_ARR[@]}" -eq 0 ]; then
  echo "ERROR: GPU_LIST is empty" >&2
  exit 2
fi

if [ "$DRY_RUN" -eq 0 ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; run this on the GPU server" >&2
    exit 2
  fi
  for gpu in "${GPU_ARR[@]}"; do
    used="$(nvidia-smi -i "$gpu" \
      --query-gpu=memory.used --format=csv,noheader,nounits \
      2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
    if ! [[ "$used" =~ ^[0-9]+$ ]]; then
      echo "ERROR: GPU $gpu is not visible to nvidia-smi" >&2
      exit 2
    fi
    if [ "${ALLOW_BUSY:-0}" != "1" ] && [ "$used" -gt "$MAX_USED_MIB" ]; then
      echo "ERROR: GPU $gpu already uses ${used} MiB" \
           "(MAX_USED_MIB=$MAX_USED_MIB)." >&2
      echo "Free it or re-run with ALLOW_BUSY=1." >&2
      exit 2
    fi
  done
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run: SACM-STC full + TPD"
  echo "  config: $CONFIG"
  echo "  arms:   $ARM"
  echo "  seeds:  ${SEED_ARR[*]}"
  echo "  gpus:   $GPU_LIST"
  echo "  runs:   ${#SEED_ARR[@]} x 800 epochs + 20-dataset eval"
  echo "  control: sacm_stc full (Dice 56.30) / sacm_official (55.40)"
  exit 0
fi

mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${TPD_FULL_LOG:-$LOG_ROOT/run_${STAMP}.log}"
PID_FILE="$LOG_ROOT/run_${STAMP}.pid"

if [ "${TPD_FULL_DAEMON:-0}" != "1" ]; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "ERROR: another full+TPD run is already active." >&2
    echo "       lock: $LOCK_FILE" >&2
    exit 2
  fi
  GPU_COUNT="${#GPU_ARR[@]}"
  RUNS=${#SEED_ARR[@]}
  WAVES=$(( (RUNS + GPU_COUNT - 1) / GPU_COUNT ))
  echo "Launching SACM-STC full + TPD."
  echo "  seeds: ${SEED_ARR[*]}"
  echo "  gpus:  $GPU_LIST"
  echo "  runs:  $RUNS ($WAVES wave(s) on $GPU_COUNT GPUs)"
  echo "  log:   $LOG"
  if command -v setsid >/dev/null 2>&1; then
    TPD_FULL_LOG="$LOG" TPD_FULL_DAEMON=1 SEEDS="${SEED_ARR[*]}" \
      GPU_LIST="$GPU_LIST" setsid nohup bash "$0" \
      >>"$LOG" 2>&1 </dev/null &
  else
    TPD_FULL_LOG="$LOG" TPD_FULL_DAEMON=1 SEEDS="${SEED_ARR[*]}" \
      GPU_LIST="$GPU_LIST" nohup bash "$0" \
      >>"$LOG" 2>&1 </dev/null &
  fi
  PID=$!
  echo "$PID" >"$PID_FILE"
  echo "  pid:   $PID"
  echo "  pid file: $PID_FILE"
  exit 0
fi

echo "============================================================"
echo " SACM-STC full + TPD"
echo " start: $(date -Is)"
echo " host:  $(hostname)"
echo " seeds: ${SEED_ARR[*]}"
echo " gpus:  $GPU_LIST"
echo " config: $CONFIG"
echo "============================================================"

set +e
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage all \
  --config "$CONFIG" --arms "$ARM" --seeds "${SEED_ARR[@]}" \
  --gpus "$GPU_LIST"
STATUS=$?
set -e

echo "============================================================"
echo " SACM-STC full + TPD finished"
echo " end:    $(date -Is)"
echo " status: $STATUS"
echo " runs:   results/runs/sacm_stc_tpd_full/tpd_full/seed_*/"
echo " eval:   results/runs/sacm_stc_tpd_full/tpd_full/seed_*/eval/"
echo "============================================================"
exit "$STATUS"
