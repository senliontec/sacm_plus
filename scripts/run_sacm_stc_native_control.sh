#!/usr/bin/env bash
# One-click native-vs-legacy topology-source control.
#
# Reuses the two completed native arms (das_ss, das_aa_mean) read-only and
# trains only the two legacy arms (legacy_ss, legacy_aa_mean).  Then runs the
# 20-dataset eval and the paired 15-dataset analysis over all four arms.
#
# The only experimental difference in the new arms is:
#   native   Stage-2 256 -> topology-resolution (current)
#   training Stage-2 256 -> 1024 -> topology-resolution (legacy)
#
# Usage (server, repo root):
#   bash scripts/run_sacm_stc_native_control.sh
#   GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_sacm_stc_native_control.sh
#   ALLOW_BUSY=1 bash scripts/run_sacm_stc_native_control.sh
#
# Env overrides:
#   GPU_LIST=0,...,7       default all eight cards
#   SEEDS="0 1 2 3"        default four seeds
#   MAX_USED_MIB=2000      fail if a card is already occupied
#   ALLOW_BUSY=1           skip the busy-GPU check
#   WANDB_MODE=online      online | offline | disabled
#   CONTROL_DRY_RUN=1      validate config/native arms and exit without GPU work
#
# Existing legacy_* directories are resumed automatically.  The native arms
# must already have completed training and 20-dataset eval; this wrapper will
# not silently retrain them.
#
# Output:
#   results/logs/sacm_stc_native_control/control_<stamp>.log
#   results/runs/sacm_stc/legacy_*/seed_*/
#   results/ablations/sacm_stc_native_control/analysis/ablation_report.md
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

CONFIG="configs/ablations/sacm_stc_native_control.yaml"
NATIVE_ARMS=(das_ss das_aa_mean)
LEGACY_ARMS=(legacy_ss legacy_aa_mean)
ARMS=("${NATIVE_ARMS[@]}" "${LEGACY_ARMS[@]}")
SEEDS=(0 1 2 3)
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
LOG_ROOT="$REPO/results/logs/sacm_stc_native_control"
LOCK_FILE="$LOG_ROOT/control.lock"

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

if [ ! -f "$REPO/$CONFIG" ]; then
  echo "ERROR: missing config: $REPO/$CONFIG" >&2
  exit 2
fi
if [ ! -f "$REPO/checkpoints/sam_vit_l_0b3195.pth" ]; then
  echo "ERROR: missing SAM ViT-L checkpoint:" \
       "$REPO/checkpoints/sam_vit_l_0b3195.pth" >&2
  exit 2
fi

# Catch a partially synced server before launching the control.
if ! grep -q -- '--topology-source' \
    "$REPO/src/baseline_zoo/adapters/sacm_stc.py"; then
  echo "ERROR: adapter is missing --topology-source." >&2
  echo "Sync src/baseline_zoo/adapters/sacm_stc.py first." >&2
  exit 2
fi
if ! grep -q 'topology-source: training' "$REPO/$CONFIG"; then
  echo "ERROR: control config has no legacy topology-source arm." >&2
  exit 2
fi

# The native arms must already be trained and evaluated on all 20 datasets.
EXPECTED_EPOCHS="$(
  "$PYTHON" -c \
    'import sys, yaml; print(int(yaml.safe_load(open(sys.argv[1]))["epochs"]))' \
    "$REPO/$CONFIG" 2>/dev/null || true
)"
if ! [[ "$EXPECTED_EPOCHS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: cannot read epochs from $CONFIG" >&2
  exit 2
fi
missing_native=()
for arm in "${NATIVE_ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_root="$REPO/results/runs/sacm_stc/$arm/seed_$seed"
    table="$run_root/eval/tables/all_metrics.csv"
    info="$run_root/run_info.json"
    if [ ! -f "$info" ] \
       || ! grep -q '"status"[[:space:]]*:[[:space:]]*"done"' "$info" \
       || ! grep -Eq "\"epochs\"[[:space:]]*:[[:space:]]*${EXPECTED_EPOCHS}([^0-9]|$)" "$info" \
       || [ ! -f "$table" ] \
       || [ "$(wc -l < "$table" 2>/dev/null || echo 0)" -lt 21 ]; then
      missing_native+=("$arm seed_$seed")
    fi
  done
done
if [ "${#missing_native[@]}" -gt 0 ]; then
  echo "ERROR: native arms are not complete:" >&2
  printf '  %s\n' "${missing_native[@]}" >&2
  echo "Run the DAS quick ablation first, e.g.:" >&2
  echo "  bash scripts/run_sacm_stc_das_quick.sh" >&2
  exit 2
fi

if [ "${CONTROL_DRY_RUN:-0}" = "1" ]; then
  echo "Dry run: native arms and control config are ready;"
  echo "would train ${LEGACY_ARMS[*]} for seeds ${SEEDS[*]} on GPUs $GPU_LIST."
  exit 0
fi

STALE_LEGACY=()
for arm in "${LEGACY_ARMS[@]}"; do
  if [ -d "$REPO/results/runs/sacm_stc/$arm" ]; then
    STALE_LEGACY+=("$arm")
  fi
done
if [ "${#STALE_LEGACY[@]}" -gt 0 ]; then
  echo "Existing legacy directories will be resumed/skipped:" \
       "${STALE_LEGACY[*]}"
  echo "Completed training and eval stages are not recomputed."
  echo "To intentionally start fresh, move/delete only those legacy_* dirs."
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
LOG="${CONTROL_LOG:-$LOG_ROOT/control_${STAMP}.log}"
PID_FILE="$LOG_ROOT/control_${STAMP}.pid"
FINGERPRINT="$(
  sha256sum \
    "$REPO/src/baseline_zoo/adapters/sacm_stc.py" \
    "$REPO/$CONFIG" \
    "$REPO/scripts/run_sacm_stc_native_control.sh" \
  | sha256sum | awk '{print $1}'
)"

# One instance only.  The parent takes the lock and the daemon child inherits
# FD 9; the child must not reopen/re-lock it.
if [ "${CONTROL_DAEMON:-0}" != "1" ]; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "ERROR: another native-control run is already running." >&2
    echo "       lock: $LOCK_FILE" >&2
    exit 2
  fi
  GPU_COUNT="${#GPU_ARR[@]}"
  NEW_RUNS=$(( ${#LEGACY_ARMS[@]} * ${#SEEDS[@]} ))
  WAVES=$(( (NEW_RUNS + GPU_COUNT - 1) / GPU_COUNT ))
  ETA_MIN=$(( WAVES * 210 ))
  echo "Launching SACM-STC native-vs-legacy control."
  echo "  native arms (reused): ${NATIVE_ARMS[*]}"
  echo "  legacy arms (trained): ${LEGACY_ARMS[*]}"
  echo "  seeds: ${SEEDS[*]}"
  echo "  gpus:  $GPU_LIST"
  echo "  new runs: $NEW_RUNS ($WAVES wave(s) on $GPU_COUNT GPUs)"
  echo "  eta:   ~$((ETA_MIN / 60))h$((ETA_MIN % 60))m (train + eval, no diagnostics)"
  echo "  fingerprint: $FINGERPRINT"
  echo "  log:   $LOG"
  if command -v setsid >/dev/null 2>&1; then
    CONTROL_LOG="$LOG" CONTROL_DAEMON=1 setsid nohup bash "$0" \
      >>"$LOG" 2>&1 </dev/null &
  else
    CONTROL_LOG="$LOG" CONTROL_DAEMON=1 nohup bash "$0" \
      >>"$LOG" 2>&1 </dev/null &
  fi
  PID=$!
  echo "$PID" >"$PID_FILE"
  echo "  pid:   $PID"
  echo "  pid file: $PID_FILE"
  exit 0
fi

echo "============================================================"
echo " SACM-STC native-vs-legacy topology control"
echo " start: $(date -Is)"
echo " host:  $(hostname)"
echo " gpus:  $GPU_LIST"
echo " native (reused): ${NATIVE_ARMS[*]}"
echo " legacy (new):    ${LEGACY_ARMS[*]}"
echo " seeds: ${SEEDS[*]}"
echo " config: $CONFIG"
echo " fingerprint: $FINGERPRINT"
echo " log:   $LOG"
echo "============================================================"

COMMON=(--config "$CONFIG" --arms "${ARMS[@]}" --seeds "${SEEDS[@]}" --gpus "$GPU_LIST")
TRAIN_STATUS=0
EVAL_STATUS=0
ANALYZE_STATUS=0

echo "[1/3] train (native arms are skipped)"
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage train "${COMMON[@]}" \
  || TRAIN_STATUS=$?
echo "[1/3] status=$TRAIN_STATUS"

echo "[2/3] 20-dataset eval (native arms are skipped)"
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage eval "${COMMON[@]}" \
  || EVAL_STATUS=$?
echo "[2/3] status=$EVAL_STATUS"

echo "[3/3] paired analysis"
bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage analyze "${COMMON[@]}" \
  || ANALYZE_STATUS=$?
echo "[3/3] status=$ANALYZE_STATUS"

echo "============================================================"
echo " SACM-STC native-vs-legacy control finished"
echo " end: $(date -Is)"
echo " status: train=$TRAIN_STATUS eval=$EVAL_STATUS analyze=$ANALYZE_STATUS"
echo " report: results/ablations/sacm_stc_native_control/analysis/ablation_report.md"
echo " runs:   results/runs/sacm_stc/legacy_*/seed_*/"
echo "============================================================"

if [ "$TRAIN_STATUS" -ne 0 ] || [ "$EVAL_STATUS" -ne 0 ] \
   || [ "$ANALYZE_STATUS" -ne 0 ]; then
  exit 1
fi
exit 0
