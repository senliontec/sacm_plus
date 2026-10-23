#!/usr/bin/env bash
# DANGER: force-kill EVERY process holding any NVIDIA GPU on this host.
# No confirmation prompt. Root is required (the script re-execs with sudo).
#
# It does NOT skip shells: fuser -k kills every PID holding /dev/nvidia*,
# including hidden workers and container helpers. All project training/eval
# processes (all users) are killed via pkill.
#
# If a GPU still shows memory with no visible PID, it attempts
# nvidia-smi --gpu-reset. GeForce cards often do not support reset; set
# REBOOT=1 to guarantee a clean state by rebooting after the kill phase.
#
# Run:
#   sudo bash scripts/force_clean_all_gpus.sh
#   REBOOT=1 sudo bash scripts/force_clean_all_gpus.sh   # last resort
set -uo pipefail

if [ "$(id -u)" -ne 0 ]; then
  exec sudo -- bash "$0" "$@"
fi

REBOOT="${REBOOT:-0}"
RESET="${RESET:-1}"

echo "== BEFORE =="
nvidia-smi || true

remaining_gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | awk 'NF && $1 ~ /^[0-9]+$/ {print $1}' | sort -u
}

for attempt in 1 2 3; do
  echo
  echo "== kill attempt $attempt =="

  # Kill project jobs for every user; they may respawn or hold GPU contexts.
  pkill -9 -f 'src/train/trainer.py|scripts/run_experiments.py|src/eval/evaluate.py|scripts/eval_sacm_x_20.sh' 2>/dev/null || true

  # Strongest generic cleanup: kill every process with an NVIDIA fd open.
  if command -v fuser >/dev/null 2>&1; then
    fuser -k -9 /dev/nvidia* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
  else
    echo "WARNING: fuser not installed; relying on nvidia-smi PIDs" >&2
  fi

  pids=$(remaining_gpu_pids) || pids=""
  if [ -n "$pids" ]; then
    echo "killing nvidia-smi PIDs: $pids"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi

  sleep 2
  if [ -z "$(remaining_gpu_pids)" ]; then
    echo "no visible GPU PIDs remain"
    break
  fi
done

# Reset only GPUs that still report memory but have no visible holder.
if [ "$RESET" = "1" ]; then
  for gpu in $(seq 0 7); do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || continue
    apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader 2>/dev/null | awk 'NF {print $1}' | sort -u) || apps=""
    if [[ "${used:-0}" =~ ^[0-9]+$ ]] && [ "$used" -gt 200 ] && [ -z "$apps" ]; then
      echo "GPU $gpu: ${used} MiB with no PID; attempting reset ..."
      nvidia-smi --gpu-reset -i "$gpu" 2>&1 || true
    fi
  done
fi

sleep 2
echo
echo "== AFTER =="
nvidia-smi || true

stuck=0
for gpu in $(seq 0 7); do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || continue
  if [[ "${used:-0}" =~ ^[0-9]+$ ]] && [ "$used" -gt 200 ]; then
    stuck=1
  fi
done

if [ "$stuck" -eq 1 ]; then
  echo
  echo "A GPU still has allocated memory. The context is stuck in the driver/container."
  if [ "$REBOOT" = "1" ]; then
    echo "REBOOT=1: rebooting now to release all GPU contexts."
    reboot
  else
    echo "Re-run with REBOOT=1 if you are authorized to reboot the host:"
    echo "  REBOOT=1 sudo bash scripts/force_clean_all_gpus.sh"
  fi
fi
