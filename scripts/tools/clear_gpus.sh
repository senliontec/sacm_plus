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

for pid in "${targets[@]}"; do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 2

remaining="$(gpu_pids)"
if [[ -n "$remaining" ]]; then
  echo "[gpu-clean] ERROR: selected GPU(s) still occupied by PIDs: $remaining" >&2
  ps -o pid=,user=,etime=,args= -p $remaining >&2 2>/dev/null || true
  echo "[gpu-clean] free them manually, or rerun with a different GPU set" >&2
  exit 1
fi

echo "[gpu-clean] selected GPU(s) cleared"
