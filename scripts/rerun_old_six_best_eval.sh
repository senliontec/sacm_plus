#!/usr/bin/env bash
# One-click best-only re-evaluation of the legacy six-arm SACM-GCD matrix.
#
# This does NOT retrain.  It:
#   1. archives the old six-arm eval/analysis (best checkpoints stay);
#   2. runs the 20-dataset eval for the six old arms on best_model.pth;
#   3. rebuilds the paired 15-dataset analysis.
#
# Use this when the 2x3 factorial is still needed.  If you only need the new
# DAS/native-control tables, use scripts/switch_to_best_only.sh instead.
#
# Usage (server, repo root):
#   bash scripts/rerun_old_six_best_eval.sh
#   bash scripts/rerun_old_six_best_eval.sh --dry-run
#   GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/rerun_old_six_best_eval.sh
#
# Env overrides:
#   GPU_LIST=0,...,7
#   SEEDS="0 1 2 3"
#   ALLOW_BUSY=1
#   MAX_USED_MIB=2000
#   RERUN_DAEMON=1       internal daemon flag; do not set manually
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

CONFIG="configs/ablations/sacm_gcd_scope_structure.yaml"
ARMS=(
  b0_selected_base
  b1_all_dense_base
  ss_selected_dense_selected_structure
  sa_selected_dense_all_structure
  b2_all_dense_selected_structure
  b3_all_dense_all_structure
)
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
LOG_ROOT="$REPO/results/logs/rerun_old_six_best_eval"
LOCK_FILE="$LOG_ROOT/rerun_old_six_best_eval.lock"
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
  echo "ERROR: $CONFIG must set eval_checkpoint: best_model.pth" >&2
  exit 2
fi
if [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "ERROR: SEEDS is empty" >&2
  exit 2
fi

missing=()
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    path="$REPO/results/runs/sacm_gcd/$arm/seed_$seed/best_model.pth"
    if [ ! -f "$path" ]; then
      missing+=("$arm seed_$seed")
    fi
  done
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "ERROR: missing best checkpoints; training cannot be repaired by eval:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 2
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run: old six arms and best checkpoints are complete."
  echo "Would archive old eval/analysis, then evaluate:"
  printf '  %s\n' "${ARMS[@]}"
  echo "  seeds: ${SEEDS[*]}"
  echo "  gpus:  $GPU_LIST"
  echo "Would run:"
  echo "  bash scripts/switch_to_best_only.sh"
  echo "  bash scripts/run_sacm_gcd_ablation.sh --stage eval --config $CONFIG --seeds ${SEEDS[*]} --gpus $GPU_LIST"
  echo "  bash scripts/run_sacm_gcd_ablation.sh --stage analyze --config $CONFIG --seeds ${SEEDS[*]}"
  exit 0
fi

IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
if [ "${#GPU_ARR[@]}" -eq 0 ]; then
  echo "ERROR: GPU_LIST is empty" >&2
  exit 2
fi
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
LOG="${RERUN_LOG:-$LOG_ROOT/rerun_${STAMP}.log}"
PID_FILE="$LOG_ROOT/rerun_${STAMP}.pid"

if [ "${RERUN_DAEMON:-0}" != "1" ]; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "ERROR: another old-six best eval is already running." >&2
    echo "       lock: $LOCK_FILE" >&2
    exit 2
  fi
  echo "Launching old-six best-only re-evaluation."
  echo "  arms:  ${ARMS[*]}"
  echo "  seeds: ${SEEDS[*]}"
  echo "  gpus:  $GPU_LIST"
  echo "  log:   $LOG"
  if command -v setsid >/dev/null 2>&1; then
    RERUN_LOG="$LOG" RERUN_DAEMON=1 setsid nohup bash "$0" \
      >>"$LOG" 2>&1 </dev/null &
  else
    RERUN_LOG="$LOG" RERUN_DAEMON=1 nohup bash "$0" \
      >>"$LOG" 2>&1 </dev/null &
  fi
  PID=$!
  echo "$PID" >"$PID_FILE"
  echo "  pid:   $PID"
  echo "  pid file: $PID_FILE"
  exit 0
fi

echo "============================================================"
echo " old-six best-only eval"
echo " start: $(date -Is)"
echo " host:  $(hostname)"
echo " arms:  ${ARMS[*]}"
echo " seeds: ${SEEDS[*]}"
echo " gpus:  $GPU_LIST"
echo "============================================================"

# Archive legacy eval/analysis first; best checkpoints and training remain.
bash "$REPO/scripts/switch_to_best_only.sh"

echo "[1/2] 20-dataset eval on best_model.pth"
bash "$REPO/scripts/run_sacm_gcd_ablation.sh" \
  --stage eval \
  --config "$CONFIG" \
  --arms "${ARMS[@]}" \
  --seeds "${SEEDS[@]}" \
  --gpus "$GPU_LIST"

echo "[2/2] paired analysis"
bash "$REPO/scripts/run_sacm_gcd_ablation.sh" \
  --stage analyze \
  --config "$CONFIG" \
  --arms "${ARMS[@]}" \
  --seeds "${SEEDS[@]}"

echo "============================================================"
echo " old-six best-only eval finished"
echo " end: $(date -Is)"
echo " report: results/ablations/sacm_gcd/analysis/ablation_report.md"
echo "============================================================"
