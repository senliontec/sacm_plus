#!/usr/bin/env bash
# One-click official-SACM reproduction: train seeds 0-3 on GPUs 4-7, then
# evaluate every seed on the 20 protocol test datasets (Dice/IoU/clDice/HD95).
#
# Usage (server, repo root):
#   bash scripts/run_sacm_official_seeds03_gpu4567.sh
#
# This uses the faithful official baseline config:
#   configs/models/sacm_official.yaml
# It intentionally preserves the upstream defects (no SAM pixel normalization,
# all-zero dense prompt, 60->800 stretched schedule).  It is the official-code
# row, not the corrected SACM+ variant.
#
# The unified runner assigns one seed per GPU.  Training must finish for all
# seeds before evaluation starts.
#
# Env overrides:
#   GPU_LIST=4,5,6,7      SEEDS="0 1 2 3"
#   RESUME=1              skip jobs whose completion marker is valid
#   SKIP_TRAIN=1          evaluate only
#   SKIP_EVAL=1           train only
#   EXPECTED_DATASETS=20  per-seed dataset-count check
#   MAX_USED_MIB=2000     fail if a selected GPU already uses more memory
#   ALLOW_BUSY=1          do not fail on busy GPUs
#   LOG_DIR=...           driver log directory
#   PYTHON=...            interpreter (server default: /home/zkyd/.../tsai)
#   WANDB_MODE=online     offline | disabled are passed through
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

MODEL="sacm_official"
VARIANT="default"
SEEDS_STR="${SEEDS:-0 1 2 3}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
RESUME="${RESUME:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
ALLOW_BUSY="${ALLOW_BUSY:-0}"
read -r -a SEED_ARGS <<< "$SEEDS_STR"

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

source "$REPO/scripts/skein/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$REPO/results/logs/sacm_official_seeds03_gpu4567_${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1

echo "============================================================"
echo " Official SACM: seeds $SEEDS_STR on GPUs $GPU_LIST"
echo " train: 800-epoch stretched official schedule"
echo " eval:  20 protocol datasets, Dice/IoU/clDice/HD95"
echo " note:  faithful official code (upstream defects preserved)"
echo " repo:  $REPO"
echo " log:   $LOG_DIR/driver.log"
echo "============================================================"

"$PYTHON" - "$REPO" "$GPU_LIST" "$SEEDS_STR" "$MAX_USED_MIB" \
  "$ALLOW_BUSY" "configs/models/sacm_official.yaml" <<'PY'
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
gpu_csv = sys.argv[2]
seeds = sys.argv[3].split()
max_used = int(sys.argv[4])
allow_busy = sys.argv[5] == '1'
config = repo / sys.argv[6]

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

checkpoint = repo / 'checkpoints/sam_vit_l_0b3195.pth'
if not checkpoint.is_file():
    sys.exit(f'ERROR: missing {checkpoint}')
if not config.is_file():
    sys.exit(f'ERROR: missing {config}')

print(f'preflight OK: GPUs {requested}, seeds {seeds}')
PY

COMMON_ARGS=(--model "$MODEL" --seeds "${SEED_ARGS[@]}" --gpus "$GPU_LIST")
if [ "$RESUME" = "1" ]; then
  COMMON_ARGS+=(--resume)
fi

if [ "$SKIP_TRAIN" != "1" ]; then
  echo "[1/2] training official SACM seeds ${SEEDS_STR} on GPUs ${GPU_LIST}"
  bash scripts/run_train.sh "${COMMON_ARGS[@]}"
else
  echo "[1/2] training skipped (SKIP_TRAIN=1)"
fi

if [ "$SKIP_EVAL" != "1" ]; then
  echo "[2/2] evaluating official SACM seeds ${SEEDS_STR} on 20 datasets"
  bash scripts/run_eval.sh "${COMMON_ARGS[@]}"
else
  echo "[2/2] evaluation skipped (SKIP_EVAL=1)"
fi

if [ "$SKIP_EVAL" != "1" ]; then
  "$PYTHON" scripts/pipeline/record_4metrics.py \
    --repo-root "$REPO" \
    --model "$MODEL" --variant "$VARIANT" \
    --seeds "${SEED_ARGS[@]}" \
    --expected-datasets "${EXPECTED_DATASETS:-20}"
fi

echo "============================================================"
echo " Done."
echo " driver log: $LOG_DIR/driver.log"
echo " per-run logs: results/runs/$MODEL/$VARIANT/seed_*/train.log"
echo "               results/runs/$MODEL/$VARIANT/seed_*/eval/eval.log"
echo " metrics:      results/runs/$MODEL/$VARIANT/seed_*/eval/tables/macro_metrics.csv"
echo " per-dataset:  results/runs/$MODEL/$VARIANT/seed_*/eval/tables/per_dataset_4metrics.csv"
echo " combined:     results/reports/${MODEL}_${VARIANT}_seed*_per_dataset.csv"
echo "               results/reports/${MODEL}_${VARIANT}_seed*_mean_std.csv/.md"
echo "============================================================"
