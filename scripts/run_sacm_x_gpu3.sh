#!/usr/bin/env bash
# SACM-X launcher pinned to GPU 3.
#
# Default: A/B diagnostic (20 epochs each, CHASEDB1 only):
#   INIT_STAGE2_FROM_STAGE1=false  -> current random Stage-2 init
#   INIT_STAGE2_FROM_STAGE1=true   -> Stage-2 warm-started from Stage-1
#
# Usage:
#   bash scripts/run_sacm_x_gpu3.sh
#   MODE=full bash scripts/run_sacm_x_gpu3.sh
#   MODE=full EPOCHS=200 INIT_STAGE2_FROM_STAGE1=false \
#     OUTPUT_ROOT=results/sacm_x_official_random bash scripts/run_sacm_x_gpu3.sh
#
# Environment:
#   GPU (default 3), MODE (diag|full), PRESET, EPOCHS, VAL_INTERVAL,
#   INIT_STAGE2_FROM_STAGE1, OUTPUT_ROOT, DATASETS, USE_WANDB, WANDB_MODE,
#   SKIP_TEST (default true: train only, no test-set evaluation),
#   DIAG_INITS (default "true": the random-init A arm has already been run;
#     use "false true" to rerun both arms),
#   CLEAN_GPU (default true), CLEAN_GPU_MODE (own|all, default own)
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

GPU="${GPU:-3}"
MODE="${MODE:-diag}"
PRESET="${PRESET:-full}"
USE_WANDB="${USE_WANDB:-true}"
MONITOR="${MONITOR:-official}"
SKIP_TEST="${SKIP_TEST:-true}"
CLEAN_GPU="${CLEAN_GPU:-true}"
CLEAN_GPU_MODE="${CLEAN_GPU_MODE:-own}"

clean_gpu() {
  local gpu="$1" mode="$2"
  local pids pid user remaining used
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | awk 'NF && $1 ~ /^[0-9]+$/ {print $1}' | sort -u) || pids=""
  if [ -z "$pids" ]; then
    echo "GPU $gpu: no compute processes"
    return 0
  fi

  echo "GPU $gpu processes:"
  # shellcheck disable=SC2086
  ps -o pid,user,etime,cmd -p $pids || true

  local own=() other=()
  for pid in $pids; do
    user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d '[:space:]')
    if [ "$user" = "$(id -un)" ]; then
      own+=("$pid")
    else
      other+=("$pid")
    fi
  done

  if [ "${#other[@]}" -gt 0 ] && [ "$mode" != "all" ]; then
    echo "ERROR: GPU $gpu has processes owned by other users: ${other[*]}" >&2
    echo "Set CLEAN_GPU_MODE=all only if you are authorized to kill them." >&2
    return 1
  fi

  local targets=()
  if [ "$mode" = "all" ]; then
    targets=($pids)
  else
    targets=("${own[@]}")
  fi
  if [ "${#targets[@]}" -eq 0 ]; then
    return 0
  fi

  echo "Killing GPU $gpu processes: ${targets[*]}"
  kill "${targets[@]}" 2>/dev/null || true
  for _ in $(seq 1 15); do
    sleep 1
    remaining=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | awk 'NF && $1 ~ /^[0-9]+$/ {print $1}' | sort -u) || remaining=""
    [ -z "$remaining" ] && break
  done

  if [ -n "$remaining" ]; then
    echo "Still present after SIGTERM; SIGKILL: $remaining"
    # shellcheck disable=SC2086
    kill -9 $remaining 2>/dev/null || true
    sleep 2
  fi

  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ') || used=0
  echo "GPU $gpu memory used after cleanup: ${used:-?} MiB"
  if [[ "${used:-0}" =~ ^[0-9]+$ ]] && [ "$used" -gt 200 ]; then
    echo "WARNING: GPU $gpu still reports >200 MiB; another process/container may be using it." >&2
    return 1
  fi
}

if command -v nvidia-smi >/dev/null 2>&1; then
  if [ "$CLEAN_GPU" = "true" ]; then
    clean_gpu "$GPU" "$CLEAN_GPU_MODE"
  fi
  echo "== GPU $GPU =="
  nvidia-smi -i "$GPU" --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
fi

case "$MODE" in
  diag)
    # Fast A/B: same training/val split; training-only by default.
    DIAG_DATASETS="${DIAG_DATASETS:-CHASEDB1}"
    DIAG_INITS="${DIAG_INITS:-true}"
    for init in $DIAG_INITS; do
      echo
      echo "== diag: INIT_STAGE2_FROM_STAGE1=$init =="
      GPU="$GPU" PRESET="$PRESET" EPOCHS="${DIAG_EPOCHS:-20}" \
        VAL_INTERVAL="${DIAG_VAL_INTERVAL:-5}" \
        MONITOR="$MONITOR" USE_WANDB="$USE_WANDB" SKIP_TEST="$SKIP_TEST" \
        INIT_STAGE2_FROM_STAGE1="$init" \
        DATASETS="$DIAG_DATASETS" \
        OUTPUT_ROOT="results/diag_sacm_x_init_${init}" \
        bash scripts/run_sacm_x_quick.sh
    done
    ;;
  full)
    INIT="${INIT_STAGE2_FROM_STAGE1:-true}"
    OUT="${OUTPUT_ROOT:-results/sacm_x_official_200}"
    echo
    echo "== full: INIT_STAGE2_FROM_STAGE1=$INIT output=$OUT =="
    GPU="$GPU" PRESET="$PRESET" EPOCHS="${EPOCHS:-200}" \
      VAL_INTERVAL="${VAL_INTERVAL:-10}" \
      MONITOR="$MONITOR" USE_WANDB="$USE_WANDB" SKIP_TEST="$SKIP_TEST" \
      INIT_STAGE2_FROM_STAGE1="$INIT" \
      OUTPUT_ROOT="$OUT" \
      DATASETS="${DATASETS:-}" \
      bash scripts/run_sacm_x_quick.sh
    ;;
  *)
    echo "ERROR: MODE must be 'diag' or 'full', got '$MODE'" >&2
    exit 2
    ;;
esac
