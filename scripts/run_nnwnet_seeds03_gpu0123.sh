#!/usr/bin/env bash
# One-click nnWNet run: train seeds 0-3 on GPUs 0-3, then evaluate every
# seed on the 20 protocol test datasets (Dice/IoU/clDice/HD95).
#
# Usage (server, repo root):
#   bash scripts/run_nnwnet_seeds03_gpu0123.sh
#
# The unified runner assigns one seed per GPU.  Training must finish for all
# seeds before evaluation starts.
#
# Env overrides:
#   GPU_LIST=0,1,2,3      SEEDS="0 1 2 3"       PROTOCOL=seeded5
#   RESUME=1              skip jobs whose completion marker is valid
#   SKIP_TRAIN=1          evaluate only
#   SKIP_EVAL=1           train only
#   DRY_RUN=1             print/validate commands without training/eval
#   EXPECTED_DATASETS=20  per-seed dataset-count check
#   MAX_USED_MIB=4000     fail if a selected GPU already uses more memory
#   MIN_FREE_GB=20        minimum free disk on the repo filesystem
#   ALLOW_BUSY=1          do not fail on busy GPUs
#   LOG_DIR=...           driver log directory
#   PYTHON=...            interpreter (server default: /home/zkyd/.../tsai)
#   WANDB_MODE=online     offline | disabled are passed through
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

MODEL="nnwnet"
VARIANT="default"
SEEDS_STR="${SEEDS:-0 1 2 3}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
PROTOCOL="${PROTOCOL:-seeded5}"
RESUME="${RESUME:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
ALLOW_BUSY="${ALLOW_BUSY:-0}"

for name in RESUME SKIP_TRAIN SKIP_EVAL DRY_RUN; do
  value="${!name}"
  case "$value" in
    0|false|False|FALSE) printf -v "$name" '%s' 0 ;;
    1|true|True|TRUE) printf -v "$name" '%s' 1 ;;
    *) echo "ERROR: $name must be 0 or 1" >&2; exit 2 ;;
  esac
done

read -r -a SEED_ARGS <<< "$SEEDS_STR"

# The 0123 and any future 4567 placements must never write the same run roots.
LOCK_DIR="$REPO/results/runs/.locks"
mkdir -p "$LOCK_DIR"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_DIR/nnwnet_default.lock"
  if ! flock -n 9; then
    echo "ERROR: another nnWNet default wrapper is already running" >&2
    exit 3
  fi
else
  LOCK_MK="$LOCK_DIR/nnwnet_default.lockdir"
  if ! mkdir "$LOCK_MK" 2>/dev/null; then
    echo "ERROR: another nnWNet default wrapper is already running" >&2
    exit 3
  fi
  trap 'rmdir "$LOCK_MK" 2>/dev/null || true' EXIT
fi

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
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

source "$REPO/scripts/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$REPO/results/logs/nnwnet_seeds03_gpu0123_${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1

echo "============================================================"
echo " nnWNet: seeds $SEEDS_STR on GPUs $GPU_LIST"
echo " train: 800 epochs, WNet2D + task-adapted InstanceNorm"
echo " eval:  20 protocol datasets, Dice/IoU/clDice/HD95"
echo " repo:  $REPO"
echo " log:   $LOG_DIR/driver.log"
echo "============================================================"

"$PYTHON" - "$REPO" "$GPU_LIST" "$SEEDS_STR" "$MAX_USED_MIB" \
  "$ALLOW_BUSY" "$MIN_FREE_GB" <<'PY'
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
gpu_csv = sys.argv[2]
seeds = sys.argv[3].split()
max_used = int(sys.argv[4])
allow_busy = sys.argv[5] == '1'
min_free_gb = float(sys.argv[6])

requested = [item.strip() for item in gpu_csv.split(',') if item.strip()]
if not requested:
    sys.exit('ERROR: GPU_LIST is empty')
if len(set(requested)) != len(requested):
    sys.exit(f'ERROR: duplicate GPU indices in GPU_LIST: {requested}')

try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,memory.used',
         '--format=csv,noheader,nounits'],
        text=True, stderr=subprocess.STDOUT)
except (OSError, subprocess.CalledProcessError) as exc:
    sys.exit(f'ERROR: nvidia-smi failed: {exc}')

available = {}
for line in out.strip().splitlines():
    parts = [item.strip() for item in line.split(',')]
    if len(parts) < 3:
        continue
    index, name, used = parts[:3]
    try:
        used_mib = int(float(used))
    except ValueError:
        used_mib = 0
    available[index] = (name, used_mib)

missing = [gpu for gpu in requested if gpu not in available]
if missing:
    print(f'ERROR: requested GPU(s) not present: {missing}', file=sys.stderr)
    print(f'Available GPU indices: '
          f'{sorted(available, key=lambda value: int(value) if value.isdigit() else value)}',
          file=sys.stderr)
    print('Set GPU_LIST=... (comma-separated) and re-run.', file=sys.stderr)
    sys.exit(2)

busy = [(gpu, available[gpu]) for gpu in requested
        if available[gpu][1] > max_used]
if busy and not allow_busy:
    print('ERROR: selected GPU(s) are already in use:', file=sys.stderr)
    for gpu, (name, used) in busy:
        print(f'  GPU {gpu} ({name}): {used} MiB', file=sys.stderr)
    try:
        procs = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,'
                           'used_memory', '--format=csv,noheader'],
            text=True, stderr=subprocess.STDOUT)
        print(procs, file=sys.stderr)
    except (OSError, subprocess.CalledProcessError):
        pass
    print('Free the GPUs or set ALLOW_BUSY=1 to override.', file=sys.stderr)
    sys.exit(3)

for seed in seeds:
    root = repo / f'data/sacm_3shot_seed{seed}'
    for name in ('protocol_manifest.csv', 'split_manifest.csv'):
        path = root / name
        if not path.is_file():
            sys.exit(f'ERROR: missing {path}')

config = repo / 'configs/models/nnwnet.yaml'
if not config.is_file():
    sys.exit(f'ERROR: missing {config}')

missing_modules = [
    module for module in (
        'nnunetv2', 'timm', 'batchgenerators',
        'dynamic_network_architectures')
    if importlib.util.find_spec(module) is None]
if missing_modules:
    sys.exit('ERROR: missing nnWNet dependencies: '
             + ', '.join(missing_modules)
             + f'\nInstall them with the runtime used by {sys.executable}')

free_gb = shutil.disk_usage(repo).free / (1024 ** 3)
if free_gb < min_free_gb:
    sys.exit(f'ERROR: only {free_gb:.1f} GiB free under {repo}; '
             f'MIN_FREE_GB={min_free_gb:g}')

print(f'preflight OK: GPUs {requested}, seeds {seeds}, '
      f'free {free_gb:.1f} GiB')
PY

COMMON_ARGS=(--model "$MODEL" --seeds "${SEED_ARGS[@]}"
             --gpus "$GPU_LIST" --protocol "$PROTOCOL")
if [ "$RESUME" = "1" ]; then
  COMMON_ARGS+=(--resume)
fi
if [ "$DRY_RUN" = "1" ]; then
  COMMON_ARGS+=(--dry-run)
fi

if [ "$SKIP_TRAIN" != "1" ]; then
  echo "[1/2] training nnWNet seeds ${SEEDS_STR} on GPUs ${GPU_LIST}"
  # nnWNet has no intra-run continuation; --resume is a completed-job
  # marker skip, not nnU-Net continue-training.
  env -u RESUME bash scripts/run_train.sh "${COMMON_ARGS[@]}"
else
  echo "[1/2] training skipped (SKIP_TRAIN=1)"
fi

if [ "$SKIP_EVAL" != "1" ]; then
  echo "[2/2] evaluating nnWNet seeds ${SEEDS_STR} on ${EXPECTED_DATASETS} datasets"
  env -u RESUME bash scripts/run_eval.sh "${COMMON_ARGS[@]}"
else
  echo "[2/2] evaluation skipped (SKIP_EVAL=1)"
fi

if [ "$SKIP_EVAL" != "1" ] && [ "$DRY_RUN" != "1" ]; then
  "$PYTHON" scripts/pipeline/record_4metrics.py \
    --repo-root "$REPO" \
    --model "$MODEL" --variant "$VARIANT" \
    --seeds "${SEED_ARGS[@]}" \
    --expected-datasets "${EXPECTED_DATASETS}"
fi

echo "============================================================"
echo " Done."
echo " driver log: $LOG_DIR/driver.log"
echo " per-run logs: results/runs/$MODEL/$VARIANT/seed_*/train.log"
echo "               results/runs/$MODEL/$VARIANT/seed_*/eval/eval.log"
echo " metrics:      results/runs/$MODEL/$VARIANT/seed_*/eval/tables/macro_metrics.csv"
echo " 4-metrics:    results/reports/${MODEL}_${VARIANT}_seeds*_mean_std.md"
