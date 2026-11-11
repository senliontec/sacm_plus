#!/usr/bin/env bash
# One-click TPD 200-epoch screening on 8 GPUs.
#
# Eight arms x four seeds = 32 runs:
#   selected_only
#   selected_image_prompt
#   selected_image_prompt_replace
#   selected_topology_quality
#   selected_tpd
#   selected_tpd_replace
#   selected_tpd_refine
#   selected_tpd_replace_refine
#
# The four native Stage-2 candidates are proposal-only; only the deployed
# candidate receives dense/topology GT loss.  No all-head supervision.
#
# Usage (server, repo root):
#   bash scripts/run_sacm_stc_tpd_200.sh
#   GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_sacm_stc_tpd_200.sh
#   bash scripts/run_sacm_stc_tpd_200.sh --dry-run
#
# Env overrides:
#   GPU_LIST=0,...,7
#   SEEDS="0 1 2 3"
#   MAX_USED_MIB=2000
#   ALLOW_BUSY=1
#   TPD_DAEMON=1   internal daemon flag; do not set manually
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

CONFIG="configs/ablations/sacm_stc_tpd_200.yaml"
ARMS=(
  selected_only
  selected_image_prompt
  selected_image_prompt_replace
  selected_topology_quality
  selected_tpd
  selected_tpd_replace
  selected_tpd_refine
  selected_tpd_replace_refine
)
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
LOG_ROOT="$REPO/results/logs/sacm_stc_tpd_200"
LOCK_FILE="$LOG_ROOT/tpd.lock"
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ ! -f "$REPO/$CONFIG" ]; then
  echo "ERROR: missing config: $REPO/$CONFIG" >&2
  exit 2
fi
if ! grep -q '^eval_checkpoint: best_model.pth$' "$REPO/$CONFIG"; then
  echo "ERROR: TPD config must set eval_checkpoint: best_model.pth" >&2
  exit 2
fi
if ! grep -q '^epochs: 200$' "$REPO/$CONFIG"; then
  echo "ERROR: TPD screening config must set epochs: 200" >&2
  exit 2
fi
if [ ! -f "$REPO/checkpoints/sam_vit_l_0b3195.pth" ]; then
  echo "ERROR: missing SAM ViT-L checkpoint:" \
       "$REPO/checkpoints/sam_vit_l_0b3195.pth" >&2
  exit 2
fi
if ! grep -q -- '--image-dense-prompt' \
    "$REPO/src/baseline_zoo/adapters/sacm_stc.py" \
   || ! grep -q -- '--topology-quality' \
    "$REPO/src/baseline_zoo/adapters/sacm_stc.py"; then
  echo "ERROR: adapter is missing TPD flags; sync the adapter and tpd.py." >&2
  exit 2
fi
if [ "${#SEEDS[@]}" -eq 0 ]; then
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
      echo "ERROR: GPU $gpu already uses ${used} MiB (MAX_USED_MIB=$MAX_USED_MIB)." >&2
      echo "Free it or re-run with ALLOW_BUSY=1." >&2
      exit 2
    fi
  done
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run: SACM-STC TPD 200"
  echo "  config: $CONFIG"
  echo "  arms:   ${ARMS[*]}"
  echo "  seeds:  ${SEEDS[*]}"
  echo "  gpus:   $GPU_LIST"
  echo "  runs:   $(( ${#ARMS[@]} * ${#SEEDS[@]} ))"
  echo "  supervision: selected-only; no all-head dense loss"
  echo "Would run train -> eval -> analyze."
  exit 0
fi

source "$REPO/scripts/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
if [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is online but no credentials were found." >&2
    echo "Set WANDB_API_KEY or use WANDB_MODE=offline/disabled." >&2
    exit 1
  fi
fi

mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${TPD_LOG:-$LOG_ROOT/tpd_${STAMP}.log}"
PID_FILE="$LOG_ROOT/tpd_${STAMP}.pid"

if [ "${TPD_DAEMON:-0}" != "1" ]; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "ERROR: another TPD screening is already running." >&2
    echo "       lock: $LOCK_FILE" >&2
    exit 2
  fi
  GPU_COUNT="${#GPU_ARR[@]}"
  RUNS=$(( ${#ARMS[@]} * ${#SEEDS[@]} ))
  WAVES=$(( (RUNS + GPU_COUNT - 1) / GPU_COUNT ))
  echo "Launching SACM-STC TPD 200 screening."
  echo "  arms:  ${ARMS[*]}"
  echo "  seeds: ${SEEDS[*]}"
  echo "  gpus:  $GPU_LIST"
  echo "  runs:  $RUNS ($WAVES wave(s) on $GPU_COUNT GPUs)"
  echo "  log:   $LOG"
  if command -v setsid >/dev/null 2>&1; then
    TPD_LOG="$LOG" TPD_DAEMON=1 setsid nohup bash "$0" \
      >>"$LOG" 2>&1 </dev/null &
  else
    TPD_LOG="$LOG" TPD_DAEMON=1 nohup bash "$0" \
      >>"$LOG" 2>&1 </dev/null &
  fi
  PID=$!
  echo "$PID" >"$PID_FILE"
  echo "  pid:   $PID"
  echo "  pid file: $PID_FILE"
  exit 0
fi

echo "============================================================"
echo " SACM-STC TPD 200 screening"
echo " start: $(date -Is)"
echo " host:  $(hostname)"
echo " arms:  ${ARMS[*]}"
echo " seeds: ${SEEDS[*]}"
echo " gpus:  $GPU_LIST"
echo " config: $CONFIG"
echo "============================================================"

COMMON=(--config "$CONFIG" --arms "${ARMS[@]}" --seeds "${SEEDS[@]}" --gpus "$GPU_LIST")
TRAIN_STATUS=0
EVAL_STATUS=0
ANALYZE_STATUS=0

echo "[1/3] train"
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage train "${COMMON[@]}" \
  || TRAIN_STATUS=$?
echo "[1/3] status=$TRAIN_STATUS"

echo "[2/3] 20-dataset eval on best_model.pth"
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage eval "${COMMON[@]}" \
  || EVAL_STATUS=$?
echo "[2/3] status=$EVAL_STATUS"

echo "[3/3] paired TPD analysis"
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage analyze "${COMMON[@]}" \
  || ANALYZE_STATUS=$?
echo "[3/3] status=$ANALYZE_STATUS"

echo "============================================================"
echo " SACM-STC TPD 200 finished"
echo " end: $(date -Is)"
echo " status: train=$TRAIN_STATUS eval=$EVAL_STATUS analyze=$ANALYZE_STATUS"
echo " report: results/ablations/sacm_stc_tpd_200/analysis/ablation_report.md"
echo " runs:   results/runs/sacm_stc_tpd_200/*/seed_*/"
echo "============================================================"

if [ "$TRAIN_STATUS" -ne 0 ] || [ "$EVAL_STATUS" -ne 0 ] \
   || [ "$ANALYZE_STATUS" -ne 0 ]; then
  exit 1
fi
exit 0
