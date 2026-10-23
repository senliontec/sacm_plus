#!/usr/bin/env bash
# One-click U-Net training on the frozen 3-shot split.
#
# Usage:
#   bash scripts/run_unet_train.sh                    # background by default
#   BACKGROUND=false bash scripts/run_unet_train.sh   # foreground
#   EPOCHS=50 bash scripts/run_unet_train.sh          # shorter run
#   USE_WANDB=false bash scripts/run_unet_train.sh    # no W&B
#   WANDB_MODE=offline bash scripts/run_unet_train.sh # local W&B logs
#   WANDB_LOG_ARTIFACT=true bash scripts/run_unet_train.sh  # upload checkpoint
#   GPU=1 bash scripts/run_unet_train.sh              # use physical GPU 1,
#                                                     # clearing it first
#   GPU=1 KILL_GPU_PROCS=false bash scripts/run_unet_train.sh  # keep other jobs
#
# Environment:
#   PYTHON, EPOCHS, BATCH_SIZE, LR, INPUT_SIZE, SEED, DEVICE, GPU,
#   WORKDIR, LOG_DIR, USE_WANDB, WANDB_MODE, WANDB_PROJECT,
#   WANDB_ENTITY, WANDB_GROUP, WANDB_NAME, WANDB_TAGS,
#   WANDB_BASE_URL, WANDB_API_KEY, WANDB_LOG_ARTIFACT, WANDB_FINISH_TIMEOUT,
#   KILL_OLD, KILL_GPU_PROCS
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
source scripts/wandb_env.sh

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"
INPUT_SIZE="${INPUT_SIZE:-512}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
WORKDIR="${WORKDIR:-workdirs/baselines/unet}"
LOG_DIR="${LOG_DIR:-logs}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"
WANDB_GROUP="${WANDB_GROUP:-baselines}"
WANDB_LOG_ARTIFACT="${WANDB_LOG_ARTIFACT:-false}"
GPU="${GPU:-}"
KILL_OLD="${KILL_OLD:-true}"
KILL_GPU_PROCS="${KILL_GPU_PROCS:-true}"
UNET_TRAIN_PATTERN='baselines/adapters/unet\.py --stage train'

# Resolve and validate the physical GPU before doing anything destructive.
# GPU=1 (preferred) or DEVICE=cuda:1 selects nvidia-smi index 1; the process is
# pinned with CUDA_VISIBLE_DEVICES so training cannot silently use another card.
GPU_INDEX=""
if [ "$DEVICE" = "cpu" ]; then
  if [ -n "$GPU" ]; then
    echo "WARNING: DEVICE=cpu ignores GPU=$GPU; no GPU cleanup will run." >&2
  fi
  KILL_GPU_PROCS=false
else
  case "$DEVICE" in
    cuda)
      GPU_INDEX="${GPU:-0}"
      ;;
    cuda:*)
      if [ -n "$GPU" ]; then
        echo "ERROR: set GPU or DEVICE=cuda:N, not both." >&2
        exit 2
      fi
      GPU_INDEX="${DEVICE#cuda:}"
      ;;
    *)
      echo "ERROR: unsupported DEVICE='$DEVICE' (use cpu, cuda, or cuda:N)." >&2
      exit 2
      ;;
  esac
  if ! [[ "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU must be a single integer index, got '$GPU_INDEX'." >&2
    exit 2
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: GPU $GPU_INDEX requested but nvidia-smi is unavailable." >&2
    exit 2
  fi
  if ! nvidia-smi -i "$GPU_INDEX" --query-gpu=index --format=csv,noheader \
      >/dev/null 2>&1; then
    echo "ERROR: GPU $GPU_INDEX does not exist. Available GPUs:" >&2
    nvidia-smi --query-gpu=index,name,memory.total,memory.used \
      --format=csv,noheader >&2 || true
    exit 2
  fi
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && \
      [ "$CUDA_VISIBLE_DEVICES" != "$GPU_INDEX" ]; then
    echo "WARNING: overriding CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES with GPU=$GPU_INDEX" >&2
  fi
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  DEVICE="cuda"
fi

# Stop a previous run before starting a new one.  Scope is deliberately limited
# to this project's U-Net training command (same Unix user): killing arbitrary
# GPU PIDs could destroy another user's job on a shared node.
kill_previous_unet() {
  if [ "$KILL_OLD" != "true" ]; then
    echo "KILL_OLD=false: keeping existing U-Net training process(es)."
    return 0
  fi
  local pids pid i
  pids="$(pgrep -u "$(id -u)" -f -- "$UNET_TRAIN_PATTERN" || true)"
  [ -z "$pids" ] && return 0

  echo "Stopping previous U-Net training process(es):"
  for pid in $pids; do
    printf '  pid %s: ' "$pid"
    tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true
    echo
  done

  # SIGTERM first so Python/CUDA can tear down cleanly; SIGKILL only on timeout.
  kill $pids 2>/dev/null || true
  for i in $(seq 1 20); do
    sleep 0.5
    pids="$(pgrep -u "$(id -u)" -f -- "$UNET_TRAIN_PATTERN" || true)"
    [ -z "$pids" ] && break
  done
  if [ -n "$pids" ]; then
    echo "SIGTERM timed out; force-killing: $pids"
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
  echo "Previous U-Net training stopped."
}

gpu_compute_apps() {
  nvidia-smi -i "$GPU_INDEX" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true
}

gpu_pids() {
  gpu_compute_apps \
    | awk -F',' 'NF {gsub(/[[:space:]]/, "", $1);
                     if ($1 ~ /^[0-9]+$/) print $1}' \
    | sort -u
}

# Free the selected GPU (default).  Only compute processes are touched; the
# script's own process tree is protected.  A PID owned by another user cannot
# be killed without permission, so abort instead of starting into an OOM.
kill_gpu_processes() {
  if [ "$KILL_GPU_PROCS" != "true" ]; then
    echo "KILL_GPU_PROCS=false: not touching GPU ${GPU_INDEX:-none}."
    return 0
  fi
  [ -z "$GPU_INDEX" ] && return 0

  local protect=" $$ $PPID " p="$PPID"
  while :; do
    p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
    case "$p" in ''|0|1) break ;; esac
    protect+=" $p"
  done

  local all_pids pids="" pid
  all_pids="$(gpu_pids)"
  for pid in $all_pids; do
    case "$protect" in
      *" $pid "*) echo "  skip own process tree: pid $pid" ;;
      *) pids+=" $pid" ;;
    esac
  done
  pids="${pids# }"
  if [ -z "$pids" ]; then
    echo "GPU $GPU_INDEX: no compute processes to stop."
    return 0
  fi

  # Refuse to kill another user's job (only reachable as root); abort instead
  # of starting into an OOM.
  local me_uid owner_uid foreign=""
  me_uid="$(id -u)"
  for pid in $pids; do
    owner_uid="$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [ -n "$owner_uid" ] && [ "$owner_uid" != "$me_uid" ]; then
      foreign+=" $pid(uid=$owner_uid)"
    fi
  done
  if [ -n "$foreign" ]; then
    echo "ERROR: GPU $GPU_INDEX has process(es) owned by another user:$foreign" >&2
    ps -o pid=,user=,args= -p $pids >&2 2>/dev/null || true
    exit 1
  fi

  echo "Stopping compute process(es) on GPU $GPU_INDEX:"
  ps -o pid=,user=,args= -p $pids 2>/dev/null | sed 's/^/  /' || true

  kill $pids 2>/dev/null || true
  local i alive=""
  for i in $(seq 1 20); do
    alive=""
    for pid in $pids; do
      kill -0 "$pid" 2>/dev/null && alive+=" $pid"
    done
    [ -z "$alive" ] && break
    sleep 0.5
  done
  alive="${alive# }"
  if [ -n "$alive" ]; then
    echo "SIGTERM timed out; force-killing:$alive"
    kill -9 $alive 2>/dev/null || true
    sleep 1
  fi

  # Confirm nvidia-smi no longer lists the original PIDs; a surviving PID
  # usually means another user owns it and this run would hit an OOM.
  local remaining="" current
  for i in $(seq 1 10); do
    remaining=""
    current="$(gpu_pids)"
    for pid in $current; do
      case " $pids " in
        *" $pid "*) remaining+=" $pid" ;;
      esac
    done
    [ -z "$remaining" ] && break
    sleep 0.5
  done
  if [ -n "$remaining" ]; then
    echo "ERROR: GPU $GPU_INDEX still has process(es):$remaining" >&2
    ps -o pid=,user=,args= -p $remaining >&2 2>/dev/null || true
    exit 1
  fi
  echo "GPU $GPU_INDEX cleared."
}

mkdir -p "$LOG_DIR" "$WORKDIR"

# Fail loudly instead of training silently without the requested W&B run.
if [ "$USE_WANDB" = "true" ] && [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled (mode=$WANDB_MODE) but no credentials were found." >&2
    echo "Expected ~/.netrc credentials for ${WANDB_BASE_URL}, or WANDB_API_KEY." >&2
    echo "Use WANDB_MODE=offline for a local run." >&2
    exit 1
  fi
fi

LOG="${_UNET_LOG:-$LOG_DIR/unet_train_$(date +%Y%m%d_%H%M%S).log}"

if [ "${BACKGROUND:-true}" = "true" ] && [ -z "${_UNET_BG_REENTRY:-}" ]; then
  nohup env _UNET_BG_REENTRY=1 _UNET_LOG="$LOG" BACKGROUND=false \
    bash "$SELF" "$@" >"$LOG" 2>&1 &
  echo "U-Net training started in background (pid $!)"
  echo "gpu: ${GPU_INDEX:-cpu}"
  echo "log: $LOG"
  echo "checkpoint: $WORKDIR/best_model.pth"
  echo "tail: tail -f $LOG"
  exit 0
fi

# Free the target GPU first: if another user's job blocks it, abort before
# stopping the previous U-Net run.
kill_gpu_processes
kill_previous_unet

echo "device=$DEVICE gpu=${GPU_INDEX:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
CMD=("$PYTHON" baselines/adapters/unet.py --stage train
     --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR"
     --input-size "$INPUT_SIZE" --seed "$SEED" --device "$DEVICE"
     --workdir "$WORKDIR")
if [ "$USE_WANDB" = "true" ]; then
  CMD+=(--use_wandb --wandb_project "$WANDB_PROJECT"
        --wandb_mode "$WANDB_MODE" --wandb_group "$WANDB_GROUP")
  [ -n "$WANDB_ENTITY" ] && CMD+=(--wandb_entity "$WANDB_ENTITY")
  [ -n "${WANDB_NAME:-}" ] && CMD+=(--wandb_name "$WANDB_NAME")
  [ -n "${WANDB_TAGS:-}" ] && CMD+=(--wandb_tags "$WANDB_TAGS")
  case "$WANDB_LOG_ARTIFACT" in
    1|true|TRUE|yes|on) CMD+=(--wandb_log_artifact) ;;
  esac
fi

echo "command: ${CMD[*]}"
echo "log: $LOG"
if [ -n "${_UNET_LOG:-}" ]; then
  # Background parent already redirects stdout/stderr to this log.
  "${CMD[@]}"
else
  "${CMD[@]}" 2>&1 | tee "$LOG"
fi
