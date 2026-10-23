#!/usr/bin/env bash
# DANGER: kill every process using an NVIDIA GPU on this host.
#
# Preview only (no kill):
#   bash scripts/clean_all_gpus.sh
#
# Kill all visible GPU users (current user's project jobs + any PIDs holding
# /dev/nvidia* that this account may kill):
#   CONFIRM=KILL_ALL_GPUS bash scripts/clean_all_gpus.sh
#
# Kill with sudo (including processes owned by other users):
#   CONFIRM=KILL_ALL_GPUS USE_SUDO=1 bash scripts/clean_all_gpus.sh
#
# If nvidia-smi still reports memory with no visible PID, the context is
# stuck in the driver/container; killing PIDs cannot release it. Reboot or
# reset the affected GPU (root required) as a last resort.
set -uo pipefail

CONFIRM="${CONFIRM:-}"
USE_SUDO="${USE_SUDO:-0}"

show_state() {
  echo "== nvidia-smi =="
  nvidia-smi || true
  echo
  echo "== holders of /dev/nvidia* =="
  for dev in /dev/nvidia[0-9]*; do
    [ -e "$dev" ] || continue
    echo "-- $dev --"
    if [ "$USE_SUDO" = "1" ]; then
      sudo fuser -v "$dev" 2>/dev/null || true
    else
      fuser -v "$dev" 2>/dev/null || true
    fi
  done
}

if [ "$CONFIRM" != "KILL_ALL_GPUS" ]; then
  echo "Refusing to kill anything without CONFIRM=KILL_ALL_GPUS."
  echo
  show_state
  echo
  echo "Set CONFIRM=KILL_ALL_GPUS to kill all GPU processes."
  exit 1
fi

echo "WARNING: killing all GPU processes on this host in 3 seconds..."
show_state
sleep 3

# 1) Kill project jobs that may respawn or keep the GPU context alive.
if [ "$USE_SUDO" = "1" ]; then
  sudo pkill -9 -f 'src/train/trainer.py|scripts/run_experiments.py|src/eval/evaluate.py' 2>/dev/null || true
else
  pkill -9 -u "$(id -u)" -f 'src/train/trainer.py|scripts/run_experiments.py|src/eval/evaluate.py' 2>/dev/null || true
fi

# 2) Kill every process holding an NVIDIA device node.
if [ "$USE_SUDO" = "1" ]; then
  sudo fuser -k /dev/nvidia* 2>/dev/null || true
else
  fuser -k /dev/nvidia* 2>/dev/null || true
fi

# 3) Also kill PIDs reported by nvidia-smi (covers processes without fuser).
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
  | awk 'NF && $1 ~ /^[0-9]+$/ {print $1}' | sort -u) || pids=""
if [ -n "$pids" ]; then
  echo "Killing nvidia-smi PIDs: $pids"
  if [ "$USE_SUDO" = "1" ]; then
    # shellcheck disable=SC2086
    sudo kill -9 $pids 2>/dev/null || true
  else
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi
fi

sleep 3
echo
echo "== after cleanup =="
nvidia-smi || true

echo
echo "If a GPU still shows memory usage without a process, the CUDA context"
echo "is stuck in the driver/container. Try (root, no processes on that GPU):"
echo "  sudo nvidia-smi --gpu-reset -i <gpu-id>"
echo "or reboot the host. A kill script cannot free a context with no PID."
