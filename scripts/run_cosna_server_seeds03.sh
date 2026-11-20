#!/usr/bin/env bash
# Server one-click: train CoSNA seeds 0-3 on GPUs 0-3, then evaluate all
# 20 protocol datasets and save per-seed + 4-seed summaries.
#
# Defaults follow the other baseline server scripts:
#   seeds 0 1 2 3 | GPUs 0,1,2,3 | EPOCHS=800 | INIT=scratch | W&B online
#
# INIT=scratch (default): train the CoSNA scaffold adapters/decoder and TCSU
#   jointly from the SAM ViT-L checkpoint only.  No BCTD or CoSNA
#   initialization checkpoint is read.
# INIT=cosna: read CoSNA-owned init checkpoints from COSNA_INIT_ROOT
#   (default checkpoints/cosna_init/seed_<n>/best_model.pth) and fit TCSU.
# INIT=bctd: legacy local initialization from BCTD source checkpoints.
#
# `EPOCHS` is the absolute final epoch target.  Scratch mode trains EPOCHS
# epochs; source modes subtract the init checkpoint epoch.
#
# Usage:
#   bash scripts/run_cosna_server_seeds03.sh
#   EPOCHS=200 bash scripts/run_cosna_server_seeds03.sh
#   SEEDS="0 1" GPUS="0,1" bash scripts/run_cosna_server_seeds03.sh
#   INIT=cosna COSNA_INIT_ROOT=checkpoints/cosna_init \
#     bash scripts/run_cosna_server_seeds03.sh
#   DRY_RUN=1 SKIP_GPU_CHECK=1 bash scripts/run_cosna_server_seeds03.sh
#
# Env:
#   SEEDS, GPUS, EPOCHS, INIT=scratch|cosna|bctd
#   COSNA_INIT_ROOT=checkpoints/cosna_init
#   RUN_ROOT, SUMMARY_DIR, RESUME=0 (0=fresh, 1=resume source checkpoints)
#   SKIP_TRAIN=0, SKIP_EVAL=0, EXPECTED_DATASETS=20
#   MAX_USED_MIB=2000, MIN_FREE_MIB=4000, ALLOW_BUSY=0
#   USE_WANDB=1, WANDB_MODE=online, WANDB_PROJECT, WANDB_ENTITY
#   DRY_RUN=0, SKIP_GPU_CHECK=0, PYTHON=...
#
# Outputs (default RUN_ROOT depends on INIT):
#   results/runs/cosna/server_scratch/seed_<n>/best_model.pth   (INIT=scratch)
#   results/runs/cosna/server_cosna/seed_<n>/best_model.pth     (INIT=cosna)
#   results/runs/cosna/server/seed_<n>/best_model.pth           (INIT=bctd)
#   <RUN_ROOT>/seed_<n>/eval/tables/all_metrics.csv
#   <RUN_ROOT>/seed_<n>/predictions/
#   results/reports/cosna_server_4seed/summary.{json,csv,md}
#   results/reports/cosna_server_seeds0_3_mean_std.{csv,md}
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SEEDS_STR="${SEEDS:-0 1 2 3}"
SEEDS_STR="${SEEDS_STR//,/ }"
GPUS_STR="${GPUS:-0,1,2,3}"
EPOCHS="${EPOCHS:-800}"
INIT="${INIT:-scratch}"
RUN_ROOT="${RUN_ROOT:-}"
SUMMARY_DIR="${SUMMARY_DIR:-results/reports/cosna_server_4seed}"
BCTD_ROOT="${BCTD_ROOT:-results/runs/topostrand/bctd200}"
SEED0_SOURCE="${SEED0_SOURCE:-results/runs/topostrand/pilot200_bnd005_h2/seed_0/best_model_fixed3.pth}"
COSNA_INIT_ROOT="${COSNA_INIT_ROOT:-checkpoints/cosna_init}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-4000}"
WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"

as_bool() {
  case "${1,,}" in
    1|true|yes|on) printf '1\n' ;;
    0|false|no|off) printf '0\n' ;;
    *) echo "ERROR: expected 0/1, got '$1'" >&2; exit 2 ;;
  esac
}
RESUME=$(as_bool "${RESUME:-0}")
SKIP_TRAIN=$(as_bool "${SKIP_TRAIN:-0}")
SKIP_EVAL=$(as_bool "${SKIP_EVAL:-0}")
ALLOW_BUSY=$(as_bool "${ALLOW_BUSY:-0}")
DRY_RUN=$(as_bool "${DRY_RUN:-0}")
SKIP_GPU_CHECK=$(as_bool "${SKIP_GPU_CHECK:-0}")
USE_WANDB=$(as_bool "${USE_WANDB:-1}")

read -r -a SEED_ARGS <<< "$SEEDS_STR"
read -r -a GPU_ARGS <<< "${GPUS_STR//,/ }"
if [ "${#SEED_ARGS[@]}" -eq 0 ]; then
  echo "ERROR: SEEDS is empty" >&2
  exit 2
fi
if [ "${#GPU_ARGS[@]}" -eq 0 ]; then
  echo "ERROR: GPUS is empty" >&2
  exit 2
fi
for seed in "${SEED_ARGS[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEEDS must be non-negative integers; got '$seed'" >&2
    exit 2
  fi
done
for gpu in "${GPU_ARGS[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPUS must be non-negative integers; got '$gpu'" >&2
    exit 2
  fi
done
if [ "$(printf '%s\n' "${SEED_ARGS[@]}" | sort -u | wc -l)" -ne "${#SEED_ARGS[@]}" ]; then
  echo "ERROR: duplicate seed in SEEDS" >&2
  exit 2
fi
if [ "$(printf '%s\n' "${GPU_ARGS[@]}" | sort -u | wc -l)" -ne "${#GPU_ARGS[@]}" ]; then
  echo "ERROR: duplicate GPU in GPUS" >&2
  exit 2
fi
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EPOCHS must be a positive integer; got '$EPOCHS'" >&2
  exit 2
fi
for name in MAX_USED_MIB MIN_FREE_MIB; do
  value="${!name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $name must be a non-negative integer; got '$value'" >&2
    exit 2
  fi
done
if ! [[ "$EXPECTED_DATASETS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EXPECTED_DATASETS must be positive; got '$EXPECTED_DATASETS'" >&2
  exit 2
fi
case "$INIT" in
  scratch|cosna|bctd) ;;
  *) echo "ERROR: INIT must be scratch, cosna, or bctd; got '$INIT'" >&2; exit 2 ;;
esac
if [ -z "$RUN_ROOT" ]; then
  case "$INIT" in
    scratch) RUN_ROOT="results/runs/cosna/server_scratch" ;;
    cosna) RUN_ROOT="results/runs/cosna/server_cosna" ;;
    *) RUN_ROOT="results/runs/cosna/server" ;;
  esac
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
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-$REPO/results/.tmp}"
mkdir -p "$TMPDIR"

abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$REPO/$1" ;;
  esac
}
RUN_ROOT_ABS=$(abspath "$RUN_ROOT")
SUMMARY_DIR_ABS=$(abspath "$SUMMARY_DIR")
BCTD_ROOT_ABS=$(abspath "$BCTD_ROOT")
SEED0_SOURCE_ABS=$(abspath "$SEED0_SOURCE")
COSNA_INIT_ROOT_ABS=$(abspath "$COSNA_INIT_ROOT")
VARIANT="$(basename "$RUN_ROOT")"

source "$REPO/scripts/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
if [ "$DRY_RUN" = 0 ] && [ "$USE_WANDB" = 1 ] \
   && [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B enabled but no credentials found; set WANDB_API_KEY," \
         "run 'wandb login', or use WANDB_MODE=offline/disabled." >&2
    exit 1
  fi
fi

if [ "$DRY_RUN" = 0 ]; then
  "$PYTHON" - "$REPO" "$GPUS_STR" "$SEEDS_STR" \
    "$MAX_USED_MIB" "$MIN_FREE_MIB" "$ALLOW_BUSY" \
    "$SKIP_TRAIN" "$RESUME" "$RUN_ROOT_ABS" \
    "$BCTD_ROOT_ABS" "$SEED0_SOURCE_ABS" "$SKIP_GPU_CHECK" \
    "$INIT" "$COSNA_INIT_ROOT_ABS" <<'PY'
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
gpus = [item.strip() for item in sys.argv[2].split(',') if item.strip()]
seeds = [int(item) for item in sys.argv[3].split()]
max_used = int(sys.argv[4])
min_free = int(sys.argv[5])
allow_busy = sys.argv[6] == '1'
skip_train = sys.argv[7] == '1'
resume = sys.argv[8] == '1'
run_root = Path(sys.argv[9])
bctd_root = Path(sys.argv[10])
seed0_source = Path(sys.argv[11])
skip_gpu = sys.argv[12] == '1'
init_mode = sys.argv[13]
cosna_init_root = Path(sys.argv[14])

if not skip_gpu:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free',
             '--format=csv,noheader,nounits'], text=True,
            stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f'ERROR: nvidia-smi failed: {exc}')
    available = {}
    for line in out.strip().splitlines():
        parts = [item.strip() for item in line.split(',')]
        if len(parts) < 4:
            continue
        try:
            available[parts[0]] = (parts[1], int(float(parts[2])),
                                   int(float(parts[3])))
        except ValueError:
            continue
    missing = [gpu for gpu in gpus if gpu not in available]
    if missing:
        sys.exit(f'ERROR: requested GPU(s) not present: {missing}; '
                 f'available={sorted(available)}')
    busy = []
    for gpu in gpus:
        name, used, free = available[gpu]
        if used > max_used or free < min_free:
            busy.append((gpu, name, used, free))
    if busy and not allow_busy:
        for gpu, name, used, free in busy:
            print(f'ERROR: GPU {gpu} ({name}): used={used} MiB, '
                  f'free={free} MiB; set ALLOW_BUSY=1 to override',
                  file=sys.stderr)
        sys.exit(3)
for seed in seeds:
    root = repo / f'data/sacm_3shot_seed{seed}'
    for name in ('protocol_manifest.csv', 'split_manifest.csv'):
        if not (root / name).is_file():
            sys.exit(f'ERROR: missing {root / name}')
    if skip_train:
        checkpoint = run_root / f'seed_{seed}' / 'best_model.pth'
        if not checkpoint.is_file():
            sys.exit(f'ERROR: SKIP_TRAIN=1 but missing {checkpoint}')
if not skip_train:
    if not (repo / 'checkpoints/sam_vit_l_0b3195.pth').is_file():
        sys.exit('ERROR: missing SAM ViT-L checkpoint')
    if init_mode == 'cosna':
        for seed in seeds:
            init = cosna_init_root / f'seed_{seed}' / 'best_model.pth'
            if not init.is_file():
                sys.exit(f'ERROR: INIT=cosna missing CoSNA init {init}; '
                         'set COSNA_INIT_ROOT to a directory with '
                         'seed_<n>/best_model.pth')
    elif init_mode == 'bctd':
        for seed in seeds:
            source = seed0_source if seed == 0 else (
                bctd_root / f'seed_{seed}' / 'best_model_fixed3.pth')
            if not source.is_file():
                sys.exit(f'ERROR: missing BCTD source checkpoint {source}')
    # INIT=scratch deliberately reads no initialization checkpoint.
print('preflight OK')
PY
fi

LOCK_DIR="$REPO/results/runs/.locks"
mkdir -p "$LOCK_DIR"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_DIR/cosna_server_seeds03.lock"
  if ! flock -n 9; then
    echo "ERROR: another CoSNA server driver is already running" >&2
    exit 3
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$REPO/results/logs/cosna_server_${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1

echo "============================================================"
echo " CoSNA server | seeds ${SEEDS_STR} | GPUs ${GPUS_STR} | ${EPOCHS} epochs"
echo " init: ${INIT}"
echo " mode: train (FORCE=$([ "$RESUME" = 1 ] && echo 0 || echo 1))"
echo " run root: ${RUN_ROOT}"
echo " summary:  ${SUMMARY_DIR}"
echo " log:      ${LOG_DIR}/driver.log"
if [ "$DRY_RUN" = 1 ]; then echo " DRY RUN"; fi
echo "============================================================"

# The lower-level driver owns per-seed initialization, training, prediction
# and canonical table generation.  It is the tested CoSNA baseline path.
env \
  MODE=train \
  INIT="$INIT" \
  SEEDS="$SEEDS_STR" \
  GPUS="$GPUS_STR" \
  RUN_ROOT="$RUN_ROOT" \
  FORCE="$([ "$RESUME" = 1 ] && echo 0 || echo 1)" \
  EPOCHS_TOTAL="$EPOCHS" \
  LOG_DIR="$LOG_DIR" \
  BCTD_ROOT="$BCTD_ROOT" \
  SEED0_SOURCE="$SEED0_SOURCE" \
  COSNA_INIT_ROOT="$COSNA_INIT_ROOT" \
  USE_WANDB="$USE_WANDB" \
  SKIP_TRAIN="$SKIP_TRAIN" \
  SKIP_EVAL="$SKIP_EVAL" \
  DRY_RUN="$DRY_RUN" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  WANDB_ENTITY="$WANDB_ENTITY" \
  PYTHON="$PYTHON" \
  bash "$REPO/scripts/run_cosna_baseline_seeds03.sh"

if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN=1: commands printed; no training/eval/summary executed."
  exit 0
fi
if [ "$SKIP_EVAL" = 1 ]; then
  echo "SKIP_EVAL=1: run finished; no test tables were generated."
  exit 0
fi

"$PYTHON" "$REPO/scripts/pipeline/summarize_baseline_seeds.py" \
  --run-root "$RUN_ROOT_ABS" \
  --seeds "${SEED_ARGS[@]}" \
  --protocol "$REPO/data/sacm_3shot_seed0/protocol_manifest.csv" \
  --out-dir "$SUMMARY_DIR_ABS"

if [ "$RUN_ROOT_ABS" = "$REPO/results/runs/cosna/$VARIANT" ]; then
  "$PYTHON" "$REPO/scripts/pipeline/record_4metrics.py" \
    --model cosna \
    --variant "$VARIANT" \
    --seeds "${SEED_ARGS[@]}" \
    --expected-datasets "$EXPECTED_DATASETS"
else
  echo "WARN: non-standard RUN_ROOT; skipping record_4metrics.py helper"
fi

echo
echo "Done."
echo "  per-seed tables: $RUN_ROOT/seed_*/eval/tables/all_metrics.csv"
echo "  4-seed summary:  $SUMMARY_DIR/summary.{json,csv,md}"
echo "  driver log:      $LOG_DIR/driver.log"
