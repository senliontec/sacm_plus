#!/usr/bin/env bash
# One-click 8-GPU DAS quick ablation.
#
# Four arms x four seeds = 16 runs, 300 epochs, train + 20-dataset eval,
# no candidate diagnostics.  On 8 free 3090s this is two waves:
#   ~2.1 h train + ~0.4 h eval per wave -> ~5.1 h total + analysis.
#
# New arm names (das_*) preserve the previous mean/best results under
# results/runs/sacm_stc.  The previous ablation is NOT overwritten.
#
# Usage (server, repo root):
#   bash scripts/run_sacm_stc_das_quick.sh
#   GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_sacm_stc_das_quick.sh
#   ALLOW_BUSY=1 bash scripts/run_sacm_stc_das_quick.sh
#   WANDB_MODE=offline bash scripts/run_sacm_stc_das_quick.sh
#
# Env overrides:
#   GPU_LIST=0,...,7       default all eight cards
#   MAX_USED_MIB=2000      fail if a card is already occupied
#   ALLOW_BUSY=1           skip the busy-GPU check
#   ALLOW_RESUME=1         allow existing das_* run directories to resume
#   WANDB_MODE=online      online | offline | disabled
#
# The eval stage refuses any arm/seed whose run_info.json is not a completed
# matching-budget training run, so a crashed training run cannot silently
# enter the 20-dataset table.
#
# Output:
#   results/logs/sacm_stc_das_quick/quick_<stamp>.log
#   results/runs/sacm_stc/das_*/seed_*/
#   results/ablations/sacm_stc_das/analysis/ablation_report.md
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

# Resolve the same interpreter as run_sacm_stc_ablation.sh before sourcing
# wandb_env.sh: the W&B credential probe needs yaml/wandb, which the system
# Python may not have.
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

CONFIG="configs/ablations/sacm_stc_das_quick.yaml"
ARMS=(
  das_ss
  das_aa_mean
  das_aa_dep
  das_as_dep
)
SEEDS=(0 1 2 3)
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
LOG_ROOT="$REPO/results/logs/sacm_stc_das_quick"

# A stale das_* directory can make --resume skip a "complete" run or reuse a
# partial checkpoint from an earlier code version.  Refuse to start unless
# the operator explicitly opts into resuming.
STALE_ARMS=()
for arm in "${ARMS[@]}"; do
  if [ -d "$REPO/results/runs/sacm_stc/$arm" ]; then
    STALE_ARMS+=("$arm")
  fi
done
if [ "${#STALE_ARMS[@]}" -gt 0 ]; then
  if [ "${ALLOW_RESUME:-0}" = "1" ]; then
    echo "WARNING: resuming existing das_* directories: ${STALE_ARMS[*]}"
  else
    echo "ERROR: stale das_* result directories already exist:" >&2
    for arm in "${STALE_ARMS[@]}"; do
      echo "  results/runs/sacm_stc/$arm" >&2
    done
    echo "Move/delete only these das_* directories, or re-run with" >&2
    echo "ALLOW_RESUME=1 if you intentionally want to resume them." >&2
    exit 2
  fi
fi
LOCK_FILE="$LOG_ROOT/quick.lock"

mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
# The daemon child inherits DAS_LOG from the parent; without this it would
# print a new, unused path in the banner while writing to the parent's log.
LOG="${DAS_LOG:-$LOG_ROOT/quick_${STAMP}.log}"
PID_FILE="$LOG_ROOT/quick_${STAMP}.pid"

if [ ! -f "$REPO/$CONFIG" ]; then
  echo "ERROR: missing config: $REPO/$CONFIG" >&2
  exit 2
fi
if [ ! -f "$REPO/checkpoints/sam_vit_l_0b3195.pth" ]; then
  echo "ERROR: missing SAM ViT-L checkpoint:" \
       "$REPO/checkpoints/sam_vit_l_0b3195.pth" >&2
  exit 2
fi
# Catch a partially synced server before launching 16 runs: the adapter must
# understand the deployment-aggregation flags, and the runner/analyzer must
# enforce the completed-training/eval guards.
if ! grep -q -- '--stage2-loss-aggregate' \
    "$REPO/src/baseline_zoo/adapters/sacm_stc.py" \
   || ! grep -q -- '--allow-untrained-eval' \
    "$REPO/scripts/pipeline/run_sacm_stc_ablation.py" \
   || ! grep -q 'require_train_complete' \
    "$REPO/scripts/pipeline/analyze_sacm_stc_ablation.py"; then
  echo "ERROR: server code is only partially synced." >&2
  echo "Sync adapter, run_sacm_stc_ablation.py and" >&2
  echo "analyze_sacm_stc_ablation.py before launching." >&2
  exit 2
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
    echo "ERROR: GPU $gpu already uses ${used} MiB" \
         "(MAX_USED_MIB=$MAX_USED_MIB)." >&2
    echo "Free it or re-run with ALLOW_BUSY=1." >&2
    exit 2
  fi
done

# W&B is default-on in this repo; fail early rather than after a long queue.
source "$REPO/scripts/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
if [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is online but no credentials were found." >&2
    echo "Set WANDB_API_KEY or use WANDB_MODE=offline/disabled." >&2
    exit 1
  fi
fi

# One instance only.  The parent takes the lock and the daemon child inherits
# FD 9; the child must not reopen/re-lock it or it can race the parent.
if [ "${DAS_DAEMON:-0}" != "1" ]; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "ERROR: another DAS quick ablation is already running." >&2
    echo "       lock: $LOCK_FILE" >&2
    exit 2
  fi
  GPU_COUNT="${#GPU_ARR[@]}"
  RUNS=$(( ${#ARMS[@]} * ${#SEEDS[@]} ))
  WAVES=$(( (RUNS + GPU_COUNT - 1) / GPU_COUNT ))
  ETA_MIN=$(( WAVES * 150 + 10 ))
  echo "Launching DAS quick ablation."
  echo "  arms:  ${ARMS[*]}"
  echo "  seeds: ${SEEDS[*]}"
  echo "  gpus:  $GPU_LIST"
  echo "  runs:  $RUNS ($WAVES wave(s) on $GPU_COUNT GPUs)"
  echo "  eta:   ~$((ETA_MIN / 60))h$((ETA_MIN % 60))m (train+eval, no diagnostics)"
  echo "  log:   $LOG"
  DAS_LOG="$LOG" DAS_DAEMON=1 nohup bash "$0" >>"$LOG" 2>&1 &
  PID=$!
  echo "$PID" >"$PID_FILE"
  echo "  pid:   $PID"
  echo "  pid file: $PID_FILE"
  exit 0
fi

echo "============================================================"
echo " DAS quick ablation"
echo " start: $(date -Is)"
echo " host:  $(hostname)"
echo " gpus:  $GPU_LIST"
echo " arms:  ${ARMS[*]}"
echo " seeds: ${SEEDS[*]}"
echo " config: $CONFIG"
echo " log:   $LOG"
echo "============================================================"

COMMON=(--config "$CONFIG" --arms "${ARMS[@]}" --seeds "${SEEDS[@]}" --gpus "$GPU_LIST")
TRAIN_STATUS=0
EVAL_STATUS=0
ANALYZE_STATUS=0

echo "[1/3] train"
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage train "${COMMON[@]}" \
  || TRAIN_STATUS=$?
echo "[1/3] status=$TRAIN_STATUS"

echo "[2/3] 20-dataset eval (best_model.pth)"
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage eval "${COMMON[@]}" \
  || EVAL_STATUS=$?
echo "[2/3] status=$EVAL_STATUS"

echo "[3/3] paired analysis"
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage analyze "${COMMON[@]}" \
  || ANALYZE_STATUS=$?
echo "[3/3] status=$ANALYZE_STATUS"

echo "============================================================"
echo " DAS quick ablation finished"
echo " end: $(date -Is)"
echo " status: train=$TRAIN_STATUS eval=$EVAL_STATUS analyze=$ANALYZE_STATUS"
echo " report: results/ablations/sacm_stc_das/analysis/ablation_report.md"
echo " runs:   results/runs/sacm_stc/das_*/seed_*/"
echo "============================================================"

if [ "$TRAIN_STATUS" -ne 0 ] || [ "$EVAL_STATUS" -ne 0 ] \
   || [ "$ANALYZE_STATUS" -ne 0 ]; then
  exit 1
fi
exit 0
