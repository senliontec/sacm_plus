#!/usr/bin/env bash
# Final SACM-STC one-click rerun: train -> eval -> analyze.
#
# This wrapper is intentionally separate from run_sacm_stc_das_quick.sh.
# It runs the frozen final protocol:
#
#   configs/ablations/sacm_stc_final.yaml
#   results/runs/sacm_stc/final_*/seed_0..3
#   results/ablations/sacm_stc_final/analysis/
#
# It does NOT run candidate diagnostics.  The ablation runner's --stage all
# path always includes diagnostics, so this wrapper drives the three stages
# explicitly.
#
# Usage (server, repo root):
#   bash scripts/run_sacm_stc_final.sh
#   FINAL_ARMS="final_full final_sel_sel" bash scripts/run_sacm_stc_final.sh
#   GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_sacm_stc_final.sh
#   ALLOW_RESUME=1 bash scripts/run_sacm_stc_final.sh
#
# Env overrides:
#   GPU_LIST=0,...,7       default all eight cards
#   SEEDS="0 1 2 3"        default four seeds
#   FINAL_ARMS="..."       comma/space separated arm subset
#   MAX_USED_MIB=2000      fail if a card is already occupied
#   ALLOW_BUSY=1           skip the busy-GPU check
#   ALLOW_RESUME=1         allow existing final_* run directories
#   ALLOW_CODE_CHANGE=1    replace a stale code/config fingerprint
#   PER_RUN_MIN=300        rough per-run minutes for the ETA display
#   WANDB_MODE=online      online | offline | disabled
#
# Output:
#   results/logs/sacm_stc_final/final_<stamp>.log
#   results/logs/sacm_stc_final/final.lock
#   results/logs/sacm_stc_final/code_fingerprint.txt
#   results/ablations/sacm_stc_final/analysis/ablation_report.md
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

CONFIG="configs/ablations/sacm_stc_final.yaml"
DEFAULT_ARMS=(
  final_full
  final_sel_sel
  final_sel_all
  final_all_sel
  final_all_all_mean
  final_sel_none
  final_all_none
)
if [ -n "${FINAL_ARMS:-}" ]; then
  IFS=', ' read -r -a ARMS <<< "$FINAL_ARMS"
else
  ARMS=("${DEFAULT_ARMS[@]}")
fi
SEEDS_STR="${SEEDS:-0 1 2 3}"
read -r -a SEED_ARR <<< "$SEEDS_STR"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
PER_RUN_MIN="${PER_RUN_MIN:-300}"
LOG_ROOT="$REPO/results/logs/sacm_stc_final"
LOCK_FILE="$LOG_ROOT/final.lock"
FINGERPRINT_FILE="$LOG_ROOT/code_fingerprint.txt"

# Resolve the same interpreter as the ablation wrapper before sourcing
# wandb_env.sh: the W&B credential probe needs yaml/wandb.
if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  elif [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
export PYTHON

if [ "${#ARMS[@]}" -eq 0 ]; then
  echo "ERROR: FINAL_ARMS is empty" >&2
  exit 2
fi
if [ "${#SEED_ARR[@]}" -eq 0 ]; then
  echo "ERROR: SEEDS is empty" >&2
  exit 2
fi
if [ ! -f "$REPO/$CONFIG" ]; then
  echo "ERROR: missing config: $REPO/$CONFIG" >&2
  exit 2
fi
if [ ! -f "$REPO/checkpoints/sam_vit_l_0b3195.pth" ]; then
  echo "ERROR: missing SAM ViT-L checkpoint:" \
       "$REPO/checkpoints/sam_vit_l_0b3195.pth" >&2
  exit 2
fi

# Catch a partially synced server before launching the final matrix.
if ! grep -q -- '--stage2-loss-aggregate' \
    "$REPO/src/baseline_zoo/adapters/sacm_stc.py" \
   || ! grep -q -- '--allow-untrained-eval' \
    "$REPO/scripts/pipeline/run_sacm_stc_ablation.py" \
   || ! grep -q 'require_eval_complete' \
    "$REPO/scripts/pipeline/analyze_sacm_stc_ablation.py"; then
  echo "ERROR: final-run code is only partially synced." >&2
  echo "Sync adapter, ablation runner and analyzer before launching." >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"

# Refuse to mix this final run with an earlier final_* directory unless the
# operator explicitly opts into resume.
STALE_ARMS=()
for arm in "${ARMS[@]}"; do
  if [ -d "$REPO/results/runs/sacm_stc/$arm" ]; then
    STALE_ARMS+=("$arm")
  fi
done
if [ "${#STALE_ARMS[@]}" -gt 0 ]; then
  if [ "${ALLOW_RESUME:-0}" = "1" ]; then
    echo "WARNING: resuming existing final_* directories: ${STALE_ARMS[*]}"
  else
    echo "ERROR: final_* result directories already exist:" >&2
    for arm in "${STALE_ARMS[@]}"; do
      echo "  results/runs/sacm_stc/$arm" >&2
    done
    echo "Move/delete only these final_* directories, or re-run with" >&2
    echo "ALLOW_RESUME=1 if you intentionally want to resume them." >&2
    exit 2
  fi
fi

# Freeze the code and config.  The per-run provenance records the complete
# source manifest; this wrapper prevents accidentally mixing code versions
# across the final matrix.
mapfile -t FP_FILES < <(
  {
    find "$REPO/src/baseline_zoo/models/sacm_stc" -type f -name '*.py' -print
    printf '%s\n' \
      "$REPO/src/baseline_zoo/adapters/sacm_stc.py" \
      "$REPO/scripts/pipeline/run_sacm_stc_ablation.py" \
      "$REPO/scripts/pipeline/analyze_sacm_stc_ablation.py" \
      "$REPO/scripts/run_sacm_stc_ablation.sh" \
      "$REPO/scripts/run_sacm_stc_final.sh" \
      "$REPO/$CONFIG"
  } | sort
)
if [ "${#FP_FILES[@]}" -eq 0 ]; then
  echo "ERROR: no files found for code fingerprint" >&2
  exit 2
fi
FINGERPRINT="$(sha256sum "${FP_FILES[@]}" | sha256sum | awk '{print $1}')"
if [ -f "$FINGERPRINT_FILE" ]; then
  PREVIOUS="$(tr -d '[:space:]' < "$FINGERPRINT_FILE")"
  if [ "$PREVIOUS" != "$FINGERPRINT" ]; then
    if [ "${ALLOW_CODE_CHANGE:-0}" = "1" ]; then
      echo "WARNING: code/config changed; replacing fingerprint $PREVIOUS -> $FINGERPRINT"
    else
      echo "ERROR: code/config changed since the recorded final fingerprint." >&2
      echo "  recorded: $PREVIOUS" >&2
      echo "  current:  $FINGERPRINT" >&2
      echo "Set ALLOW_CODE_CHANGE=1 only if this is an intentional restart." >&2
      exit 2
    fi
  fi
fi
printf '%s\n' "$FINGERPRINT" > "$FINGERPRINT_FILE"

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
    echo "ERROR: GPU $gpu already uses ${used} MiB" \
         "(MAX_USED_MIB=$MAX_USED_MIB)." >&2
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

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${FINAL_LOG:-$LOG_ROOT/final_${STAMP}.log}"
PID_FILE="$LOG_ROOT/final_${STAMP}.pid"

# One instance only.  The parent takes the lock and the daemon child inherits
# FD 9; the child must not reopen/re-lock it.
if [ "${FINAL_DAEMON:-0}" != "1" ]; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "ERROR: another SACM-STC final rerun is already running." >&2
    echo "       lock: $LOCK_FILE" >&2
    exit 2
  fi
  GPU_COUNT="${#GPU_ARR[@]}"
  RUNS=$(( ${#ARMS[@]} * ${#SEED_ARR[@]} ))
  WAVES=$(( (RUNS + GPU_COUNT - 1) / GPU_COUNT ))
  ETA_MIN=$(( WAVES * PER_RUN_MIN + 10 ))
  echo "Launching SACM-STC final rerun."
  echo "  arms:  ${ARMS[*]}"
  echo "  seeds: ${SEED_ARR[*]}"
  echo "  gpus:  $GPU_LIST"
  echo "  runs:  $RUNS ($WAVES wave(s) on $GPU_COUNT GPUs)"
  echo "  eta:   ~$((ETA_MIN / 60))h$((ETA_MIN % 60))m (train+eval, no diagnostics)"
  echo "  fingerprint: $FINGERPRINT"
  echo "  log:   $LOG"
  FINAL_LOG="$LOG" FINAL_DAEMON=1 nohup bash "$0" >>"$LOG" 2>&1 &
  PID=$!
  echo "$PID" >"$PID_FILE"
  echo "  pid:   $PID"
  echo "  pid file: $PID_FILE"
  exit 0
fi

echo "============================================================"
echo " SACM-STC final rerun"
echo " start: $(date -Is)"
echo " host:  $(hostname)"
echo " gpus:  $GPU_LIST"
echo " arms:  ${ARMS[*]}"
echo " seeds: ${SEED_ARR[*]}"
echo " config: $CONFIG"
echo " fingerprint: $FINGERPRINT"
echo " log:   $LOG"
echo "============================================================"

COMMON=(--config "$CONFIG" --arms "${ARMS[@]}" --seeds "${SEED_ARR[@]}" --gpus "$GPU_LIST")
TRAIN_STATUS=0
EVAL_STATUS=0
ANALYZE_STATUS=0

echo "[1/3] train"
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage train "${COMMON[@]}" \
  || TRAIN_STATUS=$?
echo "[1/3] status=$TRAIN_STATUS"

echo "[2/3] 20-dataset eval"
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage eval "${COMMON[@]}" \
  || EVAL_STATUS=$?
echo "[2/3] status=$EVAL_STATUS"

echo "[3/3] paired analysis"
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage analyze "${COMMON[@]}" \
  || ANALYZE_STATUS=$?
echo "[3/3] status=$ANALYZE_STATUS"

echo "============================================================"
echo " SACM-STC final rerun finished"
echo " end: $(date -Is)"
echo " status: train=$TRAIN_STATUS eval=$EVAL_STATUS analyze=$ANALYZE_STATUS"
echo " report: results/ablations/sacm_stc_final/analysis/ablation_report.md"
echo " runs:   results/runs/sacm_stc/final_*/seed_*/"
echo "============================================================"

if [ "$TRAIN_STATUS" -ne 0 ] || [ "$EVAL_STATUS" -ne 0 ] \
   || [ "$ANALYZE_STATUS" -ne 0 ]; then
  exit 1
fi
exit 0
