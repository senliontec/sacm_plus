#!/usr/bin/env bash
# Clear all NVIDIA compute processes before a one-click experiment run.
#
# Usage:
#   bash scripts/tools/clear_gpus.sh            # kill GPU compute processes
#   bash scripts/tools/clear_gpus.sh --dry-run  # list only
#
# Safety:
#   - skips this script's own process tree and all ancestors
#   - skips display servers / compositors (Xorg, Wayland, GNOME, KDE, ...)
#   - never uses sudo; PIDs owned by other users simply remain and are
#     reported as an error instead of being force-killed.
set -uo pipefail

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu-clean] nvidia-smi not found; skipping GPU cleanup"
  exit 0
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
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' \
    | awk 'NF && $1 ~ /^[0-9]+$/ {print $1}' \
    | sort -u
}

pids="$(gpu_pids)"
if [[ -z "$pids" ]]; then
  echo "[gpu-clean] all GPUs are already idle"
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
  # Desktop/browser/IDE GPU helpers show up in nvidia-smi on workstations.
  # Never kill these just to free a training GPU.
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
  echo "[gpu-clean] no killable GPU compute processes"
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
  echo "[gpu-clean] ERROR: GPU still occupied by PIDs: $remaining" >&2
  ps -o pid=,user=,etime=,args= -p $remaining >&2 2>/dev/null || true
  echo "[gpu-clean] free them manually, or rerun with a different GPU set" >&2
  exit 1
fi

echo "[gpu-clean] all GPUs cleared"
