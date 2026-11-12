#!/usr/bin/env bash
# One-click SegDINO baseline:
#   train seeds 0-3 on server GPUs 4-7, then evaluate every seed on the
#   20 protocol test datasets with Dice/IoU/clDice/HD95, and write the
#   per-dataset and combined reports.
#
# Usage (server, repo root):
#   bash scripts/run_segdino_seeds03_gpu4567.sh
#   DRY_RUN=1 bash scripts/run_segdino_seeds03_gpu4567.sh
#
# Defaults:
#   model      segdino/default
#   seeds      0 1 2 3
#   GPUs       4,5,6,7  (one seed per GPU)
#   epochs     config default: 800 (shared project budget)
#   input      512 (config entrypoint; the README's 256 is too small for
#              thin curvilinear structures)
#   weights    local DINOv3 checkpoint under checkpoints/ (server is offline)
#   metrics    dice,iou,cldice,hd95
#
# Environment overrides:
#   GPU_LIST=4,5,6,7      SEEDS="0 1 2 3"
#   DINO_CKPT=/abs/path   explicit local DINOv3 checkpoint
#   RESUME=1              skip completed train/eval jobs whose markers are valid
#   SKIP_TRAIN=1          evaluate existing checkpoints only
#   SKIP_EVAL=1           train only
#   DRY_RUN=1             print commands; no GPU jobs
#   EXPECTED_DATASETS=20  per-seed test-dataset count check
#   MAX_USED_MIB=2000     fail if a selected GPU already uses more memory
#   ALLOW_BUSY=1          allow busy GPUs
#   LOG_DIR=...           driver log directory
#   PYTHON=...            interpreter (server default: /home/zkyd/.../tsai)
#   WANDB_MODE=online     offline | disabled are passed through
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

MODEL="segdino"
VARIANT="default"
SEEDS_STR="${SEEDS:-0 1 2 3}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
RESUME="${RESUME:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_BUSY="${ALLOW_BUSY:-0}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
DINO_CKPT="${DINO_CKPT:-}"
# segdino.yaml currently defaults to the small DINOv3 backbone.  Its
# train_entrypoint does not pass --dino-size, so do not silently pair a
# different-size checkpoint with this config.
DINO_SIZE="small"

for name in RESUME SKIP_TRAIN SKIP_EVAL DRY_RUN ALLOW_BUSY; do
  value="${!name}"
  case "${value,,}" in
    1|true|yes|on) printf -v "$name" '%s' 1 ;;
    0|false|no|off) printf -v "$name" '%s' 0 ;;
    *) echo "ERROR: $name must be 0/1 (or true/false/yes/no/on/off)" >&2
       exit 2 ;;
  esac
done

read -r -a SEED_ARGS <<< "$SEEDS_STR"
if [ "${#SEED_ARGS[@]}" -eq 0 ]; then
  echo "ERROR: SEEDS is empty" >&2
  exit 2
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

# Resolve the local DINOv3 checkpoint before any GPU work.  The adapter also
# auto-detects it, but doing this here gives a clear server-side failure when
# the weight file was not uploaded.
if [ -z "$DINO_CKPT" ]; then
  default_ckpt="$REPO/checkpoints/dinov3/pytorch_model.bin"
  if [ -f "$default_ckpt" ]; then
    DINO_CKPT="$default_ckpt"
  else
    DINO_CKPT="$("$PYTHON" - "$REPO" "$DINO_SIZE" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
size = sys.argv[2]
sys.path.insert(0, str(repo / 'src'))
try:
    from baseline_zoo.models.segdino import find_dinov3_checkpoint
    path = find_dinov3_checkpoint(size, repo / 'checkpoints')
except Exception as exc:
    print(f'ERROR: cannot resolve DINOv3 checkpoint: {exc}', file=sys.stderr)
    raise SystemExit(2)
print(str(path) if path else '')
PY
)" || exit 2
  fi
fi
if [ -z "$DINO_CKPT" ] || [ ! -f "$DINO_CKPT" ]; then
  echo "ERROR: no local DINOv3 checkpoint found." >&2
  echo "Expected at least: $REPO/checkpoints/dinov3/pytorch_model.bin" >&2
  echo "Upload the matching DINOv3 small checkpoint, or set:" >&2
  echo "  DINO_CKPT=/path/to/dinov3_small.pth bash scripts/run_segdino_seeds03_gpu4567.sh" >&2
  exit 2
fi
export DINO_CKPT

source "$REPO/scripts/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$REPO/results/logs/segdino_seeds03_gpu4567_${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1

echo "============================================================"
echo " SegDINO baseline: seeds $SEEDS_STR on GPUs $GPU_LIST"
echo " train: DINOv3 small + DPT, config input 512, 800 epochs"
echo " eval:  20 protocol datasets, Dice/IoU/clDice/HD95"
echo " weights: $DINO_CKPT"
echo " repo:  $REPO"
echo " log:   $LOG_DIR/driver.log"
if [ "$DRY_RUN" = 1 ]; then
  echo " mode:  DRY RUN (no GPU jobs will be started)"
fi
echo "============================================================"

"$PYTHON" - "$REPO" "$GPU_LIST" "$SEEDS_STR" "$MAX_USED_MIB" \
  "$ALLOW_BUSY" "configs/models/segdino.yaml" "$DINO_CKPT" \
  "$EXPECTED_DATASETS" "$DRY_RUN" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
gpu_csv = sys.argv[2]
seeds = sys.argv[3].split()
max_used = int(sys.argv[4])
allow_busy = sys.argv[5] == '1'
config = repo / sys.argv[6]
dino_ckpt = Path(sys.argv[7])
expected = int(sys.argv[8])
dry_run = sys.argv[9] == '1'

requested = [item.strip() for item in gpu_csv.split(',') if item.strip()]
if not requested:
    sys.exit('ERROR: GPU_LIST is empty')
if len(set(requested)) != len(requested):
    sys.exit(f'ERROR: duplicate GPU indices in GPU_LIST: {requested}')
if not config.is_file():
    sys.exit(f'ERROR: missing config: {config}')
if not dino_ckpt.is_file():
    sys.exit(f'ERROR: missing DINOv3 checkpoint: {dino_ckpt}')

for seed in seeds:
    root = repo / f'data/sacm_3shot_seed{seed}'
    protocol = root / 'protocol_manifest.csv'
    split = root / 'split_manifest.csv'
    if not protocol.is_file() or not split.is_file():
        sys.exit(f'ERROR: missing protocol/split manifest under {root}')
    with protocol.open(newline='') as f:
        rows = list(csv.DictReader(f))
    test_datasets = {
        row['dataset'] for row in rows
        if row.get('split', '').endswith('_test')
    }
    if len(test_datasets) != expected:
        sys.exit(
            f'ERROR: seed {seed} has {len(test_datasets)} test datasets; '
            f'expected {expected}: {sorted(test_datasets)}')

if dry_run:
    print(f'preflight OK (dry-run): seeds={seeds}, '
          f'DINOv3={dino_ckpt}, expected test datasets={expected}; '
          'GPU check skipped', flush=True)
    sys.exit(0)

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

busy = [
    (gpu, available[gpu]) for gpu in requested
    if available[gpu][1] > max_used
]
if busy and not allow_busy:
    print('ERROR: selected GPU(s) are already in use:', file=sys.stderr)
    for gpu, (name, used) in busy:
        print(f'  GPU {gpu} ({name}): {used} MiB', file=sys.stderr)
    try:
        procs = subprocess.check_output(
            ['nvidia-smi',
             '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
             '--format=csv,noheader'],
            text=True, stderr=subprocess.STDOUT)
        print(procs, file=sys.stderr)
    except (OSError, subprocess.CalledProcessError):
        pass
    print('Free the GPUs or set ALLOW_BUSY=1 to override.', file=sys.stderr)
    sys.exit(3)

print(f'preflight OK: GPUs {requested}, seeds {seeds}, '
      f'DINOv3={dino_ckpt}, expected test datasets={expected}')
PY

COMMON_ARGS=(--model "$MODEL" --seeds "${SEED_ARGS[@]}" --gpus "$GPU_LIST")
if [ "$RESUME" = "1" ]; then
  COMMON_ARGS+=(--resume)
fi
if [ "$DRY_RUN" = "1" ]; then
  COMMON_ARGS+=(--dry-run)
fi

if [ "$SKIP_TRAIN" != "1" ]; then
  echo "[1/2] training SegDINO seeds ${SEEDS_STR} on GPUs ${GPU_LIST}"
  env -u RESUME bash scripts/run_train.sh "${COMMON_ARGS[@]}"
else
  echo "[1/2] training skipped (SKIP_TRAIN=1)"
fi

if [ "$SKIP_EVAL" != "1" ]; then
  if [ "$DRY_RUN" != "1" ]; then
    for seed in "${SEED_ARGS[@]}"; do
      ckpt="$REPO/results/runs/$MODEL/$VARIANT/seed_${seed}/best_model.pth"
      if [ ! -f "$ckpt" ]; then
        echo "ERROR: missing trained checkpoint: $ckpt" >&2
        echo "Training may have failed; inspect the log and re-run with RESUME=1." >&2
        exit 4
      fi
    done
  fi
  echo "[2/2] evaluating SegDINO seeds ${SEEDS_STR} on 20 datasets"
  env -u RESUME bash scripts/run_eval.sh "${COMMON_ARGS[@]}"

  if [ "$DRY_RUN" = "1" ]; then
    echo "[2/2] dry-run: skipping record_4metrics"
  else
    "$PYTHON" scripts/pipeline/record_4metrics.py \
      --repo-root "$REPO" \
      --model "$MODEL" --variant "$VARIANT" \
      --seeds "${SEED_ARGS[@]}" \
      --expected-datasets "$EXPECTED_DATASETS"
  fi
else
  echo "[2/2] evaluation skipped (SKIP_EVAL=1)"
fi

echo "============================================================"
echo " Done."
echo " driver log: $LOG_DIR/driver.log"
echo " per-run logs: results/runs/$MODEL/$VARIANT/seed_*/train.log"
echo "               results/runs/$MODEL/$VARIANT/seed_*/eval/eval.log"
echo " metrics:      results/runs/$MODEL/$VARIANT/seed_*/eval/tables/macro_metrics.csv"
echo " per-dataset:  results/runs/$MODEL/$VARIANT/seed_*/eval/tables/per_dataset_4metrics.csv"
echo " combined:     results/reports/${MODEL}_${VARIANT}_seeds0_3_per_dataset.csv"
echo "               results/reports/${MODEL}_${VARIANT}_seeds0_3_mean_std.csv/.md"
echo "============================================================"
