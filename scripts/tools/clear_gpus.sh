#!/usr/bin/env bash
# Clear compute processes on selected NVIDIA GPUs.
#
# Usage:
#   bash scripts/tools/clear_gpus.sh                 # all GPUs
#   bash scripts/tools/clear_gpus.sh --gpus 0        # only GPU 0
#   bash scripts/tools/clear_gpus.sh --gpus 0,1      # only GPUs 0 and 1
#   bash scripts/tools/clear_gpus.sh 0 1 --dry-run   # list only
#
# Safety:
#   - only processes listed on the selected GPU(s) are considered
#   - skips this script's own process tree and all ancestors
#   - skips display servers / compositors / IDE/browser GPU helpers
#   - never uses sudo; other users' PIDs cannot be killed by this user
#   - sends SIGTERM first, then SIGKILL after 5 s, and waits up to
#     GPU_CLEAN_WAIT_SECONDS (default 30) for live PIDs to disappear and GPU
#     memory to drop; <defunct> (zombie) PIDs hold no GPU resources and are
#     ignored
#   - GPU_CLEAN_MEM_LIMIT_MIB (default 1024) is the per-GPU used-memory
#     ceiling above which the script reports a stuck GPU context
set -uo pipefail

DRY_RUN=0
GPU_SPEC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --gpus)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --gpus" >&2
        exit 2
      fi
      GPU_SPEC="${2:-}"
      shift 2
      ;;
    --gpus=*)
      GPU_SPEC="${1#--gpus=}"
      shift
      ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        GPU_SPEC="${GPU_SPEC:+$GPU_SPEC,}$1"
        shift
      else
        echo "unknown argument: $1" >&2
        exit 2
      fi
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu-clean] nvidia-smi not found; skipping GPU cleanup"
  exit 0
fi

if [[ -z "$GPU_SPEC" ]]; then
  mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
      | tr -d ' '
  )
  CLEANING_ALL=1
else
  IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
  CLEANING_ALL=0
fi

# Normalize, validate, and de-duplicate the GPU indices.
normalized=()
for gpu in "${GPUS[@]}"; do
  gpu="${gpu// /}"
  [[ -z "$gpu" ]] && continue
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "[gpu-clean] ERROR: invalid GPU index: $gpu" >&2
    exit 2
  fi
  if ! nvidia-smi -i "$gpu" --query-gpu=index \
      --format=csv,noheader >/dev/null 2>&1; then
    echo "[gpu-clean] ERROR: GPU $gpu does not exist" >&2
    exit 2
  fi
  normalized+=("$gpu")
done
if [[ ${#normalized[@]} -eq 0 ]]; then
  echo "[gpu-clean] no GPU selected" >&2
  exit 2
fi
mapfile -t GPUS < <(printf '%s\n' "${normalized[@]}" | sort -n -u)

if [[ "$CLEANING_ALL" -eq 1 ]]; then
  echo "[gpu-clean] target: all GPUs (${GPUS[*]})"
else
  echo "[gpu-clean] target GPU(s): ${GPUS[*]}"
fi

get_ancestors() {
  local pid="$1"
  local ancestors=""
  while [[ -n "$pid" && "$pid" -ne 1 ]]; do
    pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    [[ -z "$pid" ]] && break
    ancestors="$ancestors $pid"
  done
  echo "$ancestors"
}

SELF_PID=$$
PROTECTED=" $SELF_PID $(get_ancestors "$SELF_PID") "

gpu_pids() {
  local gpu
  for gpu in "${GPUS[@]}"; do
    nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader 2>/dev/null
  done \
    | tr -d ' ' \
    | awk 'NF && $1 ~ /^[0-9]+$/ {print $1}' \
    | sort -u
}

pids="$(gpu_pids)"
if [[ -z "$pids" ]]; then
  echo "[gpu-clean] selected GPU(s) already idle"
  exit 0
fi

targets=()
for pid in $pids; do
  [[ -z "$pid" || "$pid" -le 2 ]] && continue
  if [[ "$PROTECTED " == *" $pid "* ]]; then
    echo "[gpu-clean] skip own process tree: pid $pid"
    continue
  fi
  pname="$(ps -p "$pid" -o comm= 2>/dev/null || true)"
  [[ -z "$pname" ]] && continue
  cmdline="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  if echo "$pname" | grep -qiE \
      'xorg|xwayland|gnome|kwin|plasma|mutter|gdm|sddm|lightdm|compositor|chrome|chromium|firefox|code|nautilus|electron|slack|discord|thunderbird'; then
    echo "[gpu-clean] skip desktop process: pid $pid ($pname)"
    continue
  fi
  if echo "$cmdline" | grep -qiE -- '--type=gpu-process|--ozone-platform|--gpu-preferences'; then
    echo "[gpu-clean] skip graphical helper: pid $pid ($pname)"
    continue
  fi
  targets+=("$pid")
done

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "[gpu-clean] no killable GPU compute processes on selected GPU(s)"
  exit 0
fi

echo "[gpu-clean] GPU compute processes to stop:"
for pid in "${targets[@]}"; do
  ps -o pid=,user=,etime=,args= -p "$pid" 2>/dev/null || true
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[gpu-clean] dry-run: nothing killed"
  exit 0
fi

# Ask the processes to exit cleanly first.  A SIGKILL delivered mid-kernel can
# leave the CUDA context stuck: nvidia-smi keeps reporting the memory as used
# even though no process is alive, and only a GPU reset/reboot recovers it.
for pid in "${targets[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done
for _ in $(seq 1 5); do
  alive=0
  for pid in "${targets[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
      break
    fi
  done
  [[ "$alive" -eq 0 ]] && break
  sleep 1
done
for pid in "${targets[@]}"; do
  kill -9 "$pid" 2>/dev/null || true
done

# A killed child stays <defunct> until its parent reaps it, and nvidia-smi
# keeps listing that dead PID.  Wait for LIVE processes to disappear and for
# the driver to release memory; zombies hold no GPU resources.
WAIT_SECONDS="${GPU_CLEAN_WAIT_SECONDS:-30}"
MEM_LIMIT_MIB="${GPU_CLEAN_MEM_LIMIT_MIB:-1024}"
deadline=$((SECONDS + WAIT_SECONDS))
live=()
zombies=()
max_used=0
while :; do
  remaining="$(gpu_pids)"
  live=()
  zombies=()
  for pid in $remaining; do
    state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [[ -z "$state" ]]; then
      # No readable state: only treat it as gone if the PID truly vanished
      # (kill -0 fails).  A cross-namespace/container PID may be invisible
      # to ps but still alive; be conservative.
      if kill -0 "$pid" 2>/dev/null; then
        live+=("$pid")
      fi
      continue
    fi
    if [[ "${state:0:1}" == "Z" ]]; then
      zombies+=("$pid")             # defunct: no GPU resources held
      continue
    fi
    live+=("$pid")
  done

  max_used=0
  for gpu in "${GPUS[@]}"; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used > max_used )); then
      max_used="$used"
    fi
  done

  if [[ ${#live[@]} -eq 0 ]]; then
    # No live process: a stale/defunct entry is harmless.  Only keep waiting
    # when a zombie is still holding GPU memory (stuck driver context).
    if [[ -z "$remaining" || "$max_used" -le "$MEM_LIMIT_MIB" ]]; then
      break
    fi
  fi
  if (( SECONDS >= deadline )); then
    break
  fi
  sleep 1
done

if [[ ${#live[@]} -gt 0 ]]; then
  echo "[gpu-clean] ERROR: selected GPU(s) still occupied by live PIDs: ${live[*]}" >&2
  ps -o pid=,user=,etime=,args= -p "${live[@]}" >&2 2>/dev/null || true
  echo "[gpu-clean] free them manually, or rerun with a different GPU set" >&2
  exit 1
fi

if [[ ${#zombies[@]} -gt 0 ]]; then
  echo "[gpu-clean] ignoring defunct (zombie) PIDs: ${zombies[*]}"
  for pid in "${zombies[@]}"; do
    ppid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
    echo "[gpu-clean]   pid $pid is <defunct> (parent ${ppid:-?}); the parent must reap it"
  done
fi

if [[ -n "$remaining" && "$max_used" -gt "$MEM_LIMIT_MIB" ]]; then
  echo "[gpu-clean] ERROR: GPU memory still in use (${max_used} MiB > ${MEM_LIMIT_MIB} MiB)." >&2
  if [[ ${#zombies[@]} -gt 0 ]]; then
    echo "[gpu-clean] a killed process left a zombie context; kill its parent launcher(s) and retry, e.g.:" >&2
    echo "[gpu-clean]   pkill -f 'scripts/sacm_x/run_experiments.py'" >&2
  fi
  echo "[gpu-clean] if memory is still held, reset the GPU (root) or reboot." >&2
  exit 1
fi

echo "[gpu-clean] selected GPU(s) cleared"
