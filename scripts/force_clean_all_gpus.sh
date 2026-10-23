#!/usr/bin/env bash
# DANGER: force-kill EVERY process holding any NVIDIA GPU on this host.
# No confirmation prompt. Root is required (the script re-execs with sudo).
#
# It kills:
#   - all project trainer/launcher/evaluator processes;
#   - every PID holding /dev/nvidia0..7 (including hidden workers/containers);
#   - every PID nvidia-smi reports.
#
# It skips only the current script, its parent, and basic shells/sshd/systemd.
# If a GPU still shows memory with no PID, it attempts nvidia-smi --gpu-reset
# for that GPU. GeForce cards may not support reset; reboot is then required.
#
# Run:
#   sudo bash scripts/force_clean_all_gpus.sh
set -uo pipefail

if [ "$(id -u)" -ne 0 ]; then
  exec sudo -- bash "$0" "$@"
fi

echo "== BEFORE =="
nvidia-smi || true

collect_pids() {
  {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true
    if command -v fuser >/dev/null 2>&1; then
      for dev in /dev/nvidia[0-9]*; do
        [ -e "$dev" ] || continue
        fuser "$dev" 2>/dev/null || true
      done
    fi
  } | tr ' ' '\n' | awk '/^[0-9]+$/ {print $1}' | sort -u
}

for attempt in 1 2 3; do
  echo
  echo "== kill attempt $attempt =="

  pkill -9 -f 'src/train/trainer.py|scripts/run_experiments.py|src/eval/evaluate.py' 2>/dev/null || true

  mapfile -t pids < <(collect_pids)
  targets=()
  for pid in "${pids[@]}"; do
    [ "$pid" = "$$" ] && continue
    [ "$pid" = "$PPID" ] && continue
    comm=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d '[:space:]')
    case "$comm" in
      sshd|systemd|init|bash|zsh|sh|Xorg)
        echo "skip $pid ($comm)"
        continue
        ;;
    esac
    targets+=("$pid")
  done

  if [ "${#targets[@]}" -eq 0 ]; then
    echo "no killable GPU PIDs remain"
    break
  fi

  echo "killing: ${targets[*]}"
  kill -9 "${targets[@]}" 2>/dev/null || true
  sleep 2
done

# Try reset only for GPUs that still report memory but have no visible holder.
if [ "${RESET:-1}" = "1" ]; then
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
echo
echo "If memory remains with no process, the GeForce driver/context is stuck;"
echo "reboot the host. A kill script cannot release a PID-less GPU context."
