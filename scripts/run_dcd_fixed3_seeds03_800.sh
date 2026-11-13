#!/usr/bin/env bash
# One-click DCD fixed3 run:
#   train SSH fixed3 base seeds 0-3 at 800 epochs
#   derive the no-fusion fixed3 checkpoint for every seed
#   evaluate fixed3 seeds 0-3 on the 20 protocol datasets
#   summarize fixed3 against the already-completed sacm_official seeds 0-3
#
# The official eval tables are read from disk; this script never trains or
# evaluates sacm_official.  It also does not train/evaluate dcd full.
#
# Default GPU placement:
#   training:  GPUs 0-3
#   fixed3 eval: GPUs 0-3 (after training; GPUs 4-7 stay free)
#
# Usage (server, repo root, preferably under tmux):
#   bash scripts/run_dcd_fixed3_seeds03_800.sh
#   DRY_RUN=1 bash scripts/run_dcd_fixed3_seeds03_800.sh
#   EVAL_GPUS=4,5,6,7 bash scripts/run_dcd_fixed3_seeds03_800.sh
#
# Env overrides:
#   SEEDS="0 1 2 3"              EPOCHS=800
#   TRAIN_GPUS=0,1,2,3           EVAL_GPUS=0,1,2,3
#   OFFICIAL_ROOT=results/runs/sacm_official/default
#   FIXED3_OUT_ROOT=results/runs/dcd_fixed3_800
#   SUMMARY_ROOT=results/ablations/dcd_fixed3_800
#   SKIP_TRAIN=0                 SKIP_EVAL=0
#   RESUME=1                     DERIVE_ONLY=0
#     RESUME=1 reuses existing complete 800-epoch SSH runs, but first checks
#     their adapter recipe with pipeline/check_ssh_recipe.py.
#     RESUME=0 retrains the base from scratch and bypasses that reuse check.
#   FORCE_DERIVE=0               FORCE_EVAL=0
#   WANDB_MODE=online            WANDB_PROJECT=sacm
#   WANDB_ENTITY=buaazqk         EVAL_WORKERS=<pipeline default>
#   EXPECTED_DATASETS=20         MIN_FREE_GB=30
#   MAX_USED_MIB=2000            ALLOW_BUSY=0
#   PYTHON=...                   LOG_DIR=...
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

# ---------------------------------------------------------------------------
# Defaults and validation
# ---------------------------------------------------------------------------
MODEL="dcd"
CONFIG="${CONFIG:-configs/ablations/dcd_ssh_full_ab.yaml}"
SEEDS_STR="${SEEDS:-0 1 2 3}"
SEEDS_STR="${SEEDS_STR//,/ }"
EPOCHS="${EPOCHS:-800}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-results/runs/sacm_official/default}"
FIXED3_OUT_ROOT="${FIXED3_OUT_ROOT:-results/runs/dcd_fixed3_800}"
SUMMARY_ROOT="${SUMMARY_ROOT:-results/ablations/dcd_fixed3_800}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"
FIXED3_WANDB_GROUP="${FIXED3_WANDB_GROUP:-dcd_fixed3_800}"

as_bool() {
  case "${1,,}" in
    1|true|yes|on) printf '1\n' ;;
    0|false|no|off) printf '0\n' ;;
    *) echo "ERROR: expected 0/1, got '$1'" >&2; exit 2 ;;
  esac
}

SKIP_TRAIN=$(as_bool "${SKIP_TRAIN:-0}")
SKIP_EVAL=$(as_bool "${SKIP_EVAL:-0}")
RESUME=$(as_bool "${RESUME:-1}")
DERIVE_ONLY=$(as_bool "${DERIVE_ONLY:-0}")
FORCE_DERIVE=$(as_bool "${FORCE_DERIVE:-0}")
FORCE_EVAL=$(as_bool "${FORCE_EVAL:-0}")
DRY_RUN=$(as_bool "${DRY_RUN:-0}")
ALLOW_BUSY=$(as_bool "${ALLOW_BUSY:-0}")
ALLOW_LOW_DISK=$(as_bool "${ALLOW_LOW_DISK:-0}")

read -r -a SEED_ARGS <<< "$SEEDS_STR"
if [ "${#SEED_ARGS[@]}" -eq 0 ]; then
  echo "ERROR: SEEDS is empty" >&2
  exit 2
fi
for seed in "${SEED_ARGS[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEEDS must be non-negative integers; got '$seed'" >&2
    exit 2
  fi
done
UNIQUE_SEEDS=$(printf '%s\n' "${SEED_ARGS[@]}" | sort -u | wc -l)
if [ "$UNIQUE_SEEDS" -ne "${#SEED_ARGS[@]}" ]; then
  echo "ERROR: SEEDS contains duplicates: ${SEED_ARGS[*]}" >&2
  exit 2
fi
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EPOCHS must be a positive integer; got '$EPOCHS'" >&2
  exit 2
fi
if ! [[ "$EXPECTED_DATASETS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EXPECTED_DATASETS must be positive" >&2
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
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-$REPO/results/.tmp}"
mkdir -p "$TMPDIR"

source "$REPO/scripts/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
if [ "$DRY_RUN" = 0 ] \
   && [ "$WANDB_MODE" != "offline" ] \
   && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled by default but no credentials were found." >&2
    echo "Set WANDB_API_KEY or run 'wandb login', or use WANDB_MODE=offline/disabled." >&2
    exit 1
  fi
fi

CONFIG_ABS="$REPO/$CONFIG"
if [[ "$CONFIG" = /* ]]; then CONFIG_ABS="$CONFIG"; fi
SAM_CHECKPOINT="$REPO/checkpoints/sam_vit_l_0b3195.pth"
if [[ "$OFFICIAL_ROOT" = /* ]]; then
  OFFICIAL_ROOT_ABS="$OFFICIAL_ROOT"
else
  OFFICIAL_ROOT_ABS="$REPO/$OFFICIAL_ROOT"
fi
if [[ "$FIXED3_OUT_ROOT" = /* ]]; then
  FIXED3_OUT_ROOT_ABS="$FIXED3_OUT_ROOT"
else
  FIXED3_OUT_ROOT_ABS="$REPO/$FIXED3_OUT_ROOT"
fi
if [[ "$SUMMARY_ROOT" = /* ]]; then
  SUMMARY_ROOT_ABS="$SUMMARY_ROOT"
else
  SUMMARY_ROOT_ABS="$REPO/$SUMMARY_ROOT"
fi

# ---------------------------------------------------------------------------
# Lock and logging
# ---------------------------------------------------------------------------
LOCK_DIR="$REPO/results/runs/.locks"
mkdir -p "$LOCK_DIR"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_DIR/dcd_fixed3_800.lock"
  if ! flock -n 9; then
    echo "ERROR: another fixed3 800-epoch driver is already running" >&2
    exit 3
  fi
else
  LOCK_MK="$LOCK_DIR/dcd_fixed3_800.lockdir"
  if ! mkdir "$LOCK_MK" 2>/dev/null; then
    echo "ERROR: another fixed3 800-epoch driver is already running" >&2
    exit 3
  fi
  trap 'rmdir "$LOCK_MK" 2>/dev/null || true' EXIT
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$REPO/results/logs/dcd_fixed3_800_${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1

echo "============================================================"
echo " DCD fixed3 | seeds ${SEEDS_STR} | ${EPOCHS} epochs"
echo " train:       GPUs ${TRAIN_GPUS}"
echo " fixed3 eval: GPUs ${EVAL_GPUS}"
echo " official:    ${OFFICIAL_ROOT_ABS} (existing eval tables, read-only)"
echo " fixed3:      head_select=fixed, deployment_head=3, head_fusion=0"
echo " no dcd full; no official retraining"
echo " repo:        ${REPO}"
echo " driver log:  ${LOG_DIR}/driver.log"
if [ "$DRY_RUN" = 1 ]; then
  echo " mode:        DRY RUN (no GPU jobs will be started)"
fi
echo "============================================================"

# ---------------------------------------------------------------------------
# Preflight: data, official eval tables, GPUs, disk
# ---------------------------------------------------------------------------
"$PYTHON" - "$REPO" "$SEEDS_STR" "$CONFIG_ABS" "$SAM_CHECKPOINT" \
  "$TRAIN_GPUS" "$EVAL_GPUS" "$MAX_USED_MIB" "$ALLOW_BUSY" "$DRY_RUN" \
  "$MIN_FREE_GB" "$ALLOW_LOW_DISK" "$OFFICIAL_ROOT_ABS" \
  "$EXPECTED_DATASETS" <<'PY'
import csv
import shutil
import subprocess
import sys
from pathlib import Path

(repo, seeds_text, config_text, sam_text, train_gpus_text, eval_gpus_text,
 max_used_text, allow_busy_text, dry_run_text, min_free_gb_text,
 allow_low_disk_text, official_root_text, expected_datasets_text) = (
    sys.argv[1:15])

repo = Path(repo).resolve()
seeds = [int(item) for item in seeds_text.split()]
config = Path(config_text)
if not config.is_absolute():
    config = repo / config
sam = Path(sam_text)
if not sam.is_absolute():
    sam = repo / sam
official_root = Path(official_root_text).resolve()
max_used = int(max_used_text)
allow_busy = allow_busy_text == '1'
dry_run = dry_run_text == '1'
min_free_gb = int(min_free_gb_text)
allow_low_disk = allow_low_disk_text == '1'
expected_datasets = int(expected_datasets_text)


def parse_gpus(text, label):
    items = [item.strip() for item in str(text).split(',') if item.strip()]
    if not items:
        raise SystemExit(f'ERROR: {label} GPU list is empty')
    if len(set(items)) != len(items):
        raise SystemExit(f'ERROR: duplicate GPU in {label}: {items}')
    return items


train_gpus = parse_gpus(train_gpus_text, 'TRAIN_GPUS')
eval_gpus = parse_gpus(eval_gpus_text, 'EVAL_GPUS')
if len(eval_gpus) < len(seeds):
    raise SystemExit(
        'ERROR: fixed3 evaluation launches one job per seed; '
        f'need at least {len(seeds)} EVAL_GPUS GPUs, got {len(eval_gpus)}')
for seed in seeds:
    root = repo / f'data/sacm_3shot_seed{seed}'
    for name in ('protocol_manifest.csv', 'split_manifest.csv'):
        path = root / name
        if not path.is_file():
            raise SystemExit(f'ERROR: missing {path}')
if not config.is_file():
    raise SystemExit(f'ERROR: missing {config}')
if not sam.is_file():
    raise SystemExit(f'ERROR: missing {sam}')

# The official comparison is read-only and must already be complete.
for seed in seeds:
    eval_root = official_root / f'seed_{seed}' / 'eval'
    for name in ('tables/all_metrics.csv', 'tables/macro_metrics.csv',
                 'run_manifest.json'):
        path = eval_root / name
        if not path.is_file():
            raise SystemExit(
                f'ERROR: missing official eval table {path}; '
                'complete sacm_official seeds 0-3 first')
    table = eval_root / 'tables' / 'all_metrics.csv'
    try:
        with open(table, newline='') as handle:
            datasets = {
                row.get('dataset') for row in csv.DictReader(handle)
                if row.get('dataset')}
    except (OSError, csv.Error) as exc:
        raise SystemExit(f'ERROR: cannot read {table}: {exc}')
    if len(datasets) < expected_datasets:
        raise SystemExit(
            f'ERROR: {table} has {len(datasets)} datasets, '
            f'expected at least {expected_datasets}')
    try:
        import json
        manifest = json.loads(
            (eval_root / 'run_manifest.json').read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(
            f'ERROR: cannot read official manifest in {eval_root}: {exc}')
    status = manifest.get('status')
    if status not in (None, 'done'):
        raise SystemExit(
            f'ERROR: official eval {eval_root} status={status!r}')

requested = sorted(set(train_gpus + eval_gpus))
if not dry_run:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.used',
             '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'ERROR: nvidia-smi failed: {exc}')
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
        raise SystemExit(
            f'ERROR: requested GPU(s) not present: {missing}; '
            f'available: {sorted(available, key=lambda x: int(x) if x.isdigit() else x)}')
    busy = [(gpu, available[gpu]) for gpu in requested
            if available[gpu][1] > max_used]
    if busy and not allow_busy:
        details = ', '.join(
            f'GPU {gpu} ({name}): {used} MiB'
            for gpu, (name, used) in busy)
        raise SystemExit(
            f'ERROR: selected GPU(s) already in use: {details}; '
            'free them or set ALLOW_BUSY=1')

free_gb = shutil.disk_usage(repo).free / (1024 ** 3)
if free_gb < min_free_gb:
    message = f'disk free {free_gb:.1f} GiB < MIN_FREE_GB={min_free_gb}'
    if allow_low_disk or dry_run:
        print(f'[preflight] WARNING: {message}', flush=True)
    else:
        raise SystemExit(
            f'ERROR: {message}; free space or set ALLOW_LOW_DISK=1')

print(
    f'preflight OK: seeds={seeds}, train_gpus={train_gpus}, '
    f'eval_gpus={eval_gpus}, official={official_root}, '
    f'free={free_gb:.1f} GiB', flush=True)
PY

"$PYTHON" -c 'import torch, yaml' >/dev/null 2>&1 || {
  echo "ERROR: ${PYTHON} cannot import torch/yaml" >&2
  exit 1
}
CONFIG_RUN_ROOT=$("$PYTHON" - "$CONFIG_ABS" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle) or {}
root = config.get('output_root')
if not root:
    raise SystemExit('ERROR: config has no output_root')
print(root)
PY
) || { echo "ERROR: cannot read output_root from ${CONFIG_ABS}" >&2; exit 1; }
if [[ "$CONFIG_RUN_ROOT" = /* ]]; then
  RUN_ROOT="$CONFIG_RUN_ROOT"
else
  RUN_ROOT="$REPO/$CONFIG_RUN_ROOT"
fi
"$PYTHON" -c 'import scipy' >/dev/null 2>&1 || \
  echo "[warn] scipy is unavailable; Wilcoxon p-values will be n/a"

if [ "$RESUME" = 1 ]; then RESUME_ARG="--resume"; else RESUME_ARG="--no-resume"; fi

# A complete 800-epoch run_info is the ablation runner's skip marker; verify
# that an existing checkpoint was really trained with this YAML recipe.
if [ "$DRY_RUN" = 0 ] \
   && { [ "$RESUME" = 1 ] || [ "$SKIP_TRAIN" = 1 ]; }; then
  echo "[0/4] checking existing SSH run recipe"
  if ! "$PYTHON" "$REPO/scripts/pipeline/check_ssh_recipe.py" \
      --run-root "$RUN_ROOT" --config "$CONFIG_ABS" --arm ssh \
      --seeds "${SEED_ARGS[@]}" --epochs "$EPOCHS"; then
    echo "ERROR: existing SSH recipe check failed; archive the affected " \
         "seed run roots or rerun with RESUME=0" >&2
    exit 1
  fi
fi

status=0

# ---------------------------------------------------------------------------
# 1. Train the SSH fixed3 base
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" = 1 ]; then
  echo "[1/4] training skipped (SKIP_TRAIN=1)"
else
  echo "[1/4] training SSH fixed3 base, seeds ${SEEDS_STR}, GPUs ${TRAIN_GPUS}"
  if [ "$DRY_RUN" = 1 ]; then
    bash "$REPO/scripts/run_dcd_ablation.sh" \
      --config "$CONFIG_ABS" --stage train --arms ssh \
      --seeds "${SEED_ARGS[@]}" --gpus "$TRAIN_GPUS" \
      --epochs "$EPOCHS" "$RESUME_ARG" --dry-run
  else
    if ! bash "$REPO/scripts/run_dcd_ablation.sh" \
        --config "$CONFIG_ABS" --stage train --arms ssh \
        --seeds "${SEED_ARGS[@]}" --gpus "$TRAIN_GPUS" \
        --epochs "$EPOCHS" "$RESUME_ARG" \
        > "$LOG_DIR/train_ssh.log" 2>&1; then
      echo "ERROR: SSH training driver failed; see ${LOG_DIR}/train_ssh.log" >&2
      status=1
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 2. Derive fixed3 checkpoints (inference-only; no retraining)
# ---------------------------------------------------------------------------
run_complete() {
  "$PYTHON" - "$1" "$EPOCHS" <<'PY'
import json
import sys
from pathlib import Path
run_root = Path(sys.argv[1])
epochs = int(sys.argv[2])
try:
    info = json.loads((run_root / 'run_info.json').read_text())
except (OSError, ValueError, TypeError):
    sys.exit(1)
try:
    recorded = int(info.get('epochs', -1))
except (TypeError, ValueError):
    sys.exit(1)
ok = (
    info.get('status') == 'done'
    and recorded == epochs
    and (run_root / 'best_model.pth').is_file()
)
sys.exit(0 if ok else 1)
PY
}

echo "[2/4] deriving fixed3 checkpoints (head_select=fixed, head=3, fusion=0)"
derived_seeds=()
for seed in "${SEED_ARGS[@]}"; do
  run_root="$RUN_ROOT/ssh/seed_${seed}"
  source_ckpt="$run_root/best_model.pth"
  output_ckpt="$run_root/best_model_fixed3_nofusion.pth"
  if [ "$DRY_RUN" = 0 ] && ! run_complete "$run_root"; then
    echo "  seed ${seed}: SSH training is not complete at ${EPOCHS} epochs; skip" >&2
    status=1
    continue
  fi
  if [ "$DRY_RUN" = 0 ] && [ ! -f "$source_ckpt" ]; then
    echo "  seed ${seed}: missing ${source_ckpt}; skip" >&2
    status=1
    continue
  fi
  if [ "$DRY_RUN" = 1 ]; then
    echo "  seed ${seed}: dry-run ${PYTHON} scripts/pipeline/make_fixed3_checkpoint.py --input ${source_ckpt} --output ${output_ckpt}"
    derived_seeds+=("$seed")
    continue
  fi
  mkdir -p "$run_root"
  if [ "$FORCE_DERIVE" = 1 ]; then
    if ! "$PYTHON" "$REPO/scripts/pipeline/make_fixed3_checkpoint.py" \
        --input "$source_ckpt" --output "$output_ckpt" --force \
        >> "$LOG_DIR/derive.log" 2>&1; then
      echo "  seed ${seed}: fixed3 derivation failed; see ${LOG_DIR}/derive.log" >&2
      status=1
      continue
    fi
  else
    if ! "$PYTHON" "$REPO/scripts/pipeline/make_fixed3_checkpoint.py" \
        --input "$source_ckpt" --output "$output_ckpt" \
        >> "$LOG_DIR/derive.log" 2>&1; then
      echo "  seed ${seed}: fixed3 derivation failed; see ${LOG_DIR}/derive.log" >&2
      status=1
      continue
    fi
  fi
  echo "  seed ${seed}: ${output_ckpt}"
  derived_seeds+=("$seed")
done
if [ "${#derived_seeds[@]}" -eq 0 ]; then
  echo "ERROR: no fixed3 checkpoint was derived; evaluation cannot start" >&2
  status=1
fi

if [ "$DERIVE_ONLY" = 1 ]; then
  echo "[2/4] DERIVE_ONLY=1; stopping after derivation"
  echo "driver log: ${LOG_DIR}/driver.log"
  exit "$status"
fi

# ---------------------------------------------------------------------------
# 3. Evaluate fixed3 on 20 datasets
# ---------------------------------------------------------------------------
launch_fixed3_eval() {
  local seed="$1"
  local gpu="$2"
  local checkpoint="$RUN_ROOT/ssh/seed_${seed}/best_model_fixed3_nofusion.pth"
  local protocol="$REPO/data/sacm_3shot_seed${seed}/protocol_manifest.csv"
  local pred_root="$FIXED3_OUT_ROOT_ABS/predictions/seed_${seed}"
  local eval_root="$FIXED3_OUT_ROOT_ABS/eval/seed_${seed}"
  local cmd=(
    "$PYTHON" "$REPO/scripts/pipeline/run_baseline_test.py"
    --method "$MODEL"
    --protocol "$protocol"
    --seed "$seed"
    --variant fixed3
    --checkpoint "$checkpoint"
    --pred-root "$pred_root"
    --paper-root "$eval_root"
    --device "cuda:${gpu}"
    --use_wandb true
    --wandb_project "$WANDB_PROJECT"
    --wandb_entity "$WANDB_ENTITY"
    --wandb_group "$FIXED3_WANDB_GROUP"
    --wandb_name "paper_metrics-dcd-fixed3-seed${seed}"
    --wandb_tags "dcd,fixed3,${EPOCHS}ep,paper"
    --metrics "dice,iou,cldice,hd95"
    --skip-visualize
  )
  if [ -n "${EVAL_WORKERS:-}" ]; then
    cmd+=(--workers "$EVAL_WORKERS")
  fi
  if [ "$FORCE_EVAL" = 1 ] || [ "$RESUME" = 0 ]; then
    cmd+=(--no-resume)
  fi
  if [ "$DRY_RUN" = 1 ]; then
    printf '[fixed3-eval]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  mkdir -p "$FIXED3_OUT_ROOT_ABS/predictions" \
           "$FIXED3_OUT_ROOT_ABS/eval" "$LOG_DIR/fixed3_eval"
  "${cmd[@]}" > "$LOG_DIR/fixed3_eval/seed${seed}.log" 2>&1 &
  echo "$!"
}

read -r -a EVAL_GPU_ARR <<< "${EVAL_GPUS//,/ }"
if [ "${#EVAL_GPU_ARR[@]}" -eq 0 ]; then
  echo "ERROR: EVAL_GPUS is empty" >&2
  exit 2
fi

echo "[3/4] evaluating fixed3 on 20 datasets, GPUs ${EVAL_GPUS}"
fixed3_pids=()
fixed3_seed_list=()
if [ "$SKIP_EVAL" = 1 ]; then
  echo "  evaluation skipped (SKIP_EVAL=1)"
else
  idx=0
  for seed in "${derived_seeds[@]}"; do
    gpu="${EVAL_GPU_ARR[$((idx % ${#EVAL_GPU_ARR[@]}))]}"
    if [ "$DRY_RUN" = 1 ]; then
      launch_fixed3_eval "$seed" "$gpu" || status=1
    else
      pid=$(launch_fixed3_eval "$seed" "$gpu") || pid=""
      if [ -z "$pid" ]; then
        echo "ERROR: could not launch fixed3 eval for seed ${seed}" >&2
        status=1
      else
        fixed3_pids+=("$pid")
        fixed3_seed_list+=("$seed")
      fi
    fi
    idx=$((idx + 1))
  done
  for i in "${!fixed3_pids[@]}"; do
    seed="${fixed3_seed_list[$i]}"
    if ! wait "${fixed3_pids[$i]}"; then
      echo "ERROR: fixed3 eval failed for seed ${seed}; " \
           "see ${LOG_DIR}/fixed3_eval/seed${seed}.log" >&2
      status=1
    fi
  done
fi

# ---------------------------------------------------------------------------
# 4. fixed3 vs sacm_official summary (CPU-only; official tables are read-only)
# ---------------------------------------------------------------------------
if [ "$SKIP_EVAL" = 1 ]; then
  echo "[4/4] summary skipped (SKIP_EVAL=1)"
elif [ "$DRY_RUN" = 1 ]; then
  echo "[4/4] dry-run summary command:"
  echo "  ${PYTHON} scripts/pipeline/summarize_fixed3_seeds.py --fixed3-root ${FIXED3_OUT_ROOT_ABS} --official-root ${OFFICIAL_ROOT_ABS} --seeds ${SEED_ARGS[*]} --epochs ${EPOCHS} --expected-datasets ${EXPECTED_DATASETS} --output-root ${SUMMARY_ROOT_ABS}"
else
  echo "[4/4] summarizing fixed3 vs sacm_official"
  mkdir -p "$SUMMARY_ROOT_ABS"
  summary_args=(
    "$PYTHON" "$REPO/scripts/pipeline/summarize_fixed3_seeds.py"
    --fixed3-root "$FIXED3_OUT_ROOT_ABS"
    --official-root "$OFFICIAL_ROOT_ABS"
    --seeds "${SEED_ARGS[@]}"
    --epochs "$EPOCHS"
    --expected-datasets "$EXPECTED_DATASETS"
    --output-root "$SUMMARY_ROOT_ABS"
  )
  if "${summary_args[@]}" >> "$LOG_DIR/summary.log" 2>&1; then
    echo "  ${SUMMARY_ROOT_ABS}/summary.md"
  else
    echo "ERROR: summary failed; see ${LOG_DIR}/summary.log" >&2
    status=1
  fi
fi

# ---------------------------------------------------------------------------
# Final status
# ---------------------------------------------------------------------------
echo "============================================================"
if [ "$status" -eq 0 ]; then
  echo " DONE"
else
  echo " FINISHED WITH FAILURES (exit ${status})"
fi
echo " driver log:        ${LOG_DIR}/driver.log"
echo " train log:         ${LOG_DIR}/train_ssh.log"
echo " ablation logs:     ${REPO}/results/logs/dcd_ablation/dcd_ssh_full_ab/"
echo " SSH runs:          ${RUN_ROOT}/ssh/seed_*/"
echo " fixed3 ckpts:      ${RUN_ROOT}/ssh/seed_*/best_model_fixed3_nofusion.pth"
echo " fixed3 eval:       ${FIXED3_OUT_ROOT_ABS}/eval/seed_*/"
echo " fixed3 eval logs:  ${LOG_DIR}/fixed3_eval/seed*.log"
echo " official (read):   ${OFFICIAL_ROOT_ABS}/seed_*/eval/"
if [ "$SKIP_EVAL" = 0 ] && [ "$DRY_RUN" = 0 ]; then
  echo " fixed3 vs official: ${SUMMARY_ROOT_ABS}/summary.md"
fi
echo " W&B train group:   dcd_ssh_full_ab"
echo " W&B fixed3 group:  ${FIXED3_WANDB_GROUP}"
echo "============================================================"
exit "$status"
