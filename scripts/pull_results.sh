#!/usr/bin/env bash
# Incrementally pull server-side results/runs into the local repo.
#
# Default pull set is analysis data (no weights / predictions):
#   eval/**, metrics_history.json, run_info.json, checkpoint.json,
#   train_config.json, val_metrics_by_dataset.csv, wandb_run.json,
#   train.log, eval.log, provenance/**
#
# The script uses rsync include/exclude filters directly; it does not run a
# remote find.  It never uses rsync --delete and never overwrites newer local
# files.  It is safe to run repeatedly while the server is still training or
# evaluating.
#
# Usage:
#   bash scripts/pull_results.sh
#   bash scripts/pull_results.sh --models "dcd:all_dense sacm_official:default" --seeds "0 1"
#   bash scripts/pull_results.sh --watch 300
#   bash scripts/pull_results.sh --until-complete --interval 300
#   bash scripts/pull_results.sh --status
#   bash scripts/pull_results.sh --with-predictions --datasets "DRIVE CREMI"
#   bash scripts/pull_results.sh --with-checkpoints
#   bash scripts/pull_results.sh --dry-run
#   bash scripts/pull_results.sh --require-complete
#
# Env overrides:
#   SERVER=zkyd@172.16.1.7
#   TARGET=/home/zkyd/code/sacm-plus
#   EXPECTED_DATASETS=20
#   PYTHON=/path/to/python
#
# Notes:
#   * --with-predictions can pull many gigabytes; use --datasets to limit it.
#   * --with-checkpoints pulls best_model.pth / stage*_best.pth only.
#   * --until-complete exits when every selected model/variant/seed has an
#     eval/tables/all_metrics.csv with EXPECTED_DATASETS rows.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
SERVER="${SERVER:-zkyd@172.16.1.7}"
TARGET="${TARGET:-/home/zkyd/code/sacm-plus}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"

# DCD has one canonical main variant, all_dense.  The historical
# default/ tree was only an eval alias of the all_dense checkpoint; it is
# archived by scripts/archive_dcd_main.sh and must not be pulled or
# presented as a separate method.
DEFAULT_MODELS=(dcd:all_dense sacm_official:default unet:default cs2net:default maskvsc:default)
DEFAULT_SEEDS=(0 1 2 3)

MODEL_FILTERS=()
SEED_FILTERS=()
DATASET_FILTERS=()
WITH_PREDICTIONS=0
WITH_CHECKPOINTS=0
DRY_RUN=0
REQUIRE_COMPLETE=0
STATUS_ONLY=0
WATCH=0
INTERVAL=300
TIMEOUT=0
RETRIES=3

usage() {
  cat <<'EOF'
Pull server-side results/runs into the local repo (analysis data, no weights).

Usage:
  bash scripts/pull_results.sh [options]

Options:
  --models LIST        model or model:variant list; default: all models
                       e.g. "dcd:all_dense sacm_official:default"
  --seeds LIST         seed list; default: all seeds
  --datasets LIST      only with --with-predictions; dataset filter
  --with-predictions   also pull predictions/ (can be very large)
  --with-checkpoints   also pull best_model.pth / stage*_best.pth
  --status             only print local run status/metrics; do not pull
  --dry-run            show what rsync would transfer; do not write
  --watch [SECONDS]    pull repeatedly (default interval 300s)
  --until-complete     repeat until selected runs have 20 dataset rows
  --interval SECONDS   watch interval (default 300)
  --timeout SECONDS    stop watch after this many seconds (0 = never)
  --retries N          rsync retry attempts on failure (default 3)
  --require-complete   exit 1 if any selected run is missing/partial
  -h, --help           show this help

Env: SERVER, TARGET, EXPECTED_DATASETS, PYTHON
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --models)
      [ $# -ge 2 ] || { echo "missing value for --models" >&2; exit 2; }
      shift
      MODEL_FILTERS=(${1//,/ })
      ;;
    --seeds)
      [ $# -ge 2 ] || { echo "missing value for --seeds" >&2; exit 2; }
      shift
      SEED_FILTERS=(${1//,/ })
      ;;
    --datasets)
      [ $# -ge 2 ] || { echo "missing value for --datasets" >&2; exit 2; }
      shift
      DATASET_FILTERS=(${1//,/ })
      ;;
    --with-predictions) WITH_PREDICTIONS=1 ;;
    --with-checkpoints) WITH_CHECKPOINTS=1 ;;
    --status) STATUS_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --require-complete) REQUIRE_COMPLETE=1 ;;
    --until-complete) REQUIRE_COMPLETE=1; WATCH=1 ;;
    --interval)
      [ $# -ge 2 ] || { echo "missing value for --interval" >&2; exit 2; }
      shift; INTERVAL="$1"
      ;;
    --timeout)
      [ $# -ge 2 ] || { echo "missing value for --timeout" >&2; exit 2; }
      shift; TIMEOUT="$1"
      ;;
    --retries)
      [ $# -ge 2 ] || { echo "missing value for --retries" >&2; exit 2; }
      shift; RETRIES="$1"
      ;;
    --watch)
      WATCH=1
      if [ $# -ge 2 ] && [[ "$2" =~ ^[0-9]+$ ]]; then
        INTERVAL="$2"
        shift
      fi
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$WITH_PREDICTIONS" -eq 1 ] && [ ${#DATASET_FILTERS[@]} -eq 0 ]; then
  echo "WARNING: --with-predictions without --datasets pulls every prediction PNG." >&2
fi
if [ "$WITH_CHECKPOINTS" -eq 1 ]; then
  echo "WARNING: --with-checkpoints pulls model weights and can be large." >&2
fi
if ! [[ "$RETRIES" =~ ^[0-9]+$ ]] || [ "$RETRIES" -lt 1 ]; then
  echo "ERROR: --retries must be a positive integer" >&2
  exit 2
fi

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  elif [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python3
  fi
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/sacm-pull.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
EXPECTED="$TMP/expected.txt"

CONTROL_DIR="${TMPDIR:-/tmp}/sacm-plus-ssh-$(id -u)"
mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"
CONTROL_PATH="$CONTROL_DIR/%C"
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$CONTROL_PATH"
          -o ControlPersist=600 -o ServerAliveInterval=30
          -o ServerAliveCountMax=4)
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

if [ "$STATUS_ONLY" -eq 0 ] && \
   ! ssh "${SSH_OPTS[@]}" "$SERVER" "test -d '$TARGET/results/runs'" >/dev/null 2>&1; then
  echo "ERROR: cannot access $SERVER:$TARGET/results/runs" >&2
  echo "Check SERVER/TARGET and network/SSH access." >&2
  exit 3
fi

pull_files() {
  local model_patterns=()
  local seed_patterns=()
  if [ ${#MODEL_FILTERS[@]} -eq 0 ] || [ "${MODEL_FILTERS[0]}" = "all" ]; then
    model_patterns=('*/*')
  else
    local spec model variant
    for spec in "${MODEL_FILTERS[@]}"; do
      if [[ "$spec" == *:* ]]; then
        model="${spec%%:*}"
        variant="${spec#*:}"
      else
        model="$spec"
        variant='*'
      fi
      model_patterns+=("$model/$variant")
    done
  fi
  if [ ${#SEED_FILTERS[@]} -eq 0 ] || [ "${SEED_FILTERS[0]}" = "all" ]; then
    seed_patterns=('*')
  else
    seed_patterns=("${SEED_FILTERS[@]}")
  fi

  local filters=()
  # Prune the large directories before the generic directory include.
  if [ "$WITH_PREDICTIONS" -eq 0 ]; then
    filters+=(--exclude='predictions/')
  fi
  if [ "$WITH_CHECKPOINTS" -eq 0 ]; then
    filters+=(--exclude='checkpoints/')
    filters+=(--exclude='*.pth')
  fi
  filters+=(--include='*/')

  local mp sp prefix ds
  for mp in "${model_patterns[@]}"; do
    for sp in "${seed_patterns[@]}"; do
      prefix="$mp/seed_$sp"
      filters+=(--include="${prefix}/eval/***")
      filters+=(--include="${prefix}/metrics_history.json")
      filters+=(--include="${prefix}/run_info.json")
      filters+=(--include="${prefix}/checkpoint.json")
      filters+=(--include="${prefix}/train_config.json")
      filters+=(--include="${prefix}/val_metrics_by_dataset.csv")
      filters+=(--include="${prefix}/wandb_run.json")
      filters+=(--include="${prefix}/train.log")
      filters+=(--include="${prefix}/eval.log")
      filters+=(--include="${prefix}/provenance/***")
      if [ "$WITH_PREDICTIONS" -eq 1 ]; then
        if [ ${#DATASET_FILTERS[@]} -eq 0 ]; then
          filters+=(--include="${prefix}/predictions/***")
        else
          for ds in "${DATASET_FILTERS[@]}"; do
            filters+=(--include="${prefix}/predictions/*/${ds}/***")
          done
        fi
      fi
    done
  done
  if [ "$WITH_CHECKPOINTS" -eq 1 ]; then
    filters+=(--include='*/best_model.pth')
    filters+=(--include='*/stage1_best.pth')
    filters+=(--include='*/stage2_best.pth')
  fi
  filters+=(--exclude='*')

  echo "=== $(date '+%F %T') pulling analysis results from $SERVER:$TARGET/results/runs ==="
  local rsync_args=(-az --partial --update --prune-empty-dirs --timeout=120
                    --info=progress2 -e "$RSYNC_SSH")
  if [ "$DRY_RUN" -eq 1 ]; then
    rsync_args+=(--dry-run)
  fi
  local attempt=1
  while :; do
    if rsync "${rsync_args[@]}" "${filters[@]}" \
        "$SERVER:$TARGET/results/runs/" "$REPO/results/runs/"; then
      return 0
    fi
    if [ "$attempt" -ge "$RETRIES" ]; then
      echo "ERROR: rsync failed after $RETRIES attempt(s)" >&2
      return 1
    fi
    attempt=$((attempt + 1))
    echo "rsync failed; retry $attempt/$RETRIES in 10s ..." >&2
    sleep 10
  done
}

check_coverage() {
  local models=("${MODEL_FILTERS[@]}")
  local seeds=("${SEED_FILTERS[@]}")
  if [ ${#models[@]} -eq 0 ] || [ "${models[0]}" = "all" ]; then
    models=("${DEFAULT_MODELS[@]}")
  fi
  if [ ${#seeds[@]} -eq 0 ] || [ "${seeds[0]}" = "all" ]; then
    seeds=("${DEFAULT_SEEDS[@]}")
  fi
  : > "$EXPECTED"
  local spec model variant seed
  for spec in "${models[@]}"; do
    if [[ "$spec" == *:* ]]; then
      model="${spec%%:*}"
      variant="${spec#*:}"
    else
      model="$spec"
      if [ "$model" = "dcd" ]; then
        variant="all_dense"
      else
        variant="default"
      fi
    fi
    for seed in "${seeds[@]}"; do
      printf '%s|%s|%s\n' "$model" "$variant" "$seed" >> "$EXPECTED"
    done
  done
  "$PYTHON" - "$REPO" "$EXPECTED_DATASETS" "$EXPECTED" <<'PY'
import csv
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
expected_n = int(sys.argv[2])
expected_path = Path(sys.argv[3])
expected = [line.strip().split('|')
            for line in expected_path.read_text().splitlines()
            if line.strip()]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def macro(rows, key):
    values = [float(row[key]) for row in rows
              if row.get(key) not in (None, '')]
    return sum(values) / len(values) if values else float('nan')


print('')
print(f'Local run status (eval/tables/all_metrics.csv; expected {expected_n} rows/run):')
ok_all = True
for model, variant, seed in expected:
    run = repo / 'results' / 'runs' / model / variant / f'seed_{seed}'
    info = {}
    info.update(read_json(run / 'checkpoint.json'))
    info.update(read_json(run / 'run_info.json'))
    train_state = info.get('status', '-')
    best_epoch = info.get('best_epoch')
    best_epoch = '-' if best_epoch is None else str(best_epoch)
    label = f'{model}/{variant} seed_{seed}'

    # Flag runs whose evaluated checkpoint actually lives under another
    # variant (e.g. a copied/aliased run) so the same weights are not
    # silently counted twice in a comparison.
    ckpt = str(info.get('best_checkpoint') or info.get('checkpoint') or '')
    parts = ckpt.split('/')
    alias = ''
    if 'runs' in parts:
        i = parts.index('runs')
        if len(parts) > i + 3 and parts[i + 1] == model and parts[i + 2] != variant:
            alias = f' alias_of={parts[i + 2]}'

    path = run / 'eval' / 'tables' / 'all_metrics.csv'
    if not path.is_file():
        print(f'  {label}: eval MISSING | train={train_state} '
              f'best_ep={best_epoch}{alias}')
        ok_all = False
        continue
    with path.open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row.get('dataset')]
    n = len(rows)
    state = 'OK' if n >= expected_n else 'PARTIAL'
    if n < expected_n:
        ok_all = False
    dice = macro(rows, 'dice_mean')
    iou = macro(rows, 'iou_mean')
    cldice = macro(rows, 'cldice_mean')
    hd95 = macro(rows, 'hd95_mean')
    print(f'  {label}: {n}/{expected_n} {state} | train={train_state} '
          f'best_ep={best_epoch}{alias} | Dice {100 * dice:.2f} '
          f'IoU {100 * iou:.2f} clDice {100 * cldice:.2f} HD95 {hd95:.2f}')
print('coverage:', 'COMPLETE' if ok_all else 'INCOMPLETE')
raise SystemExit(0 if ok_all else 1)
PY
}

if [ "$STATUS_ONLY" -eq 1 ]; then
  if check_coverage; then
    exit 0
  fi
  exit 1
fi

run_once() {
  pull_files || return 1
  if check_coverage; then
    return 0
  fi
  return 1
}

if [ "$DRY_RUN" -eq 0 ] && [ "$WATCH" -eq 1 ]; then
  START="$(date +%s)"
  while :; do
    if run_once; then
      COMPLETE=1
    else
      COMPLETE=0
    fi
    if [ "$REQUIRE_COMPLETE" -eq 1 ] && [ "$COMPLETE" -eq 1 ]; then
      echo "all selected runs are complete"
      break
    fi
    if [ "$TIMEOUT" -gt 0 ] && [ $(( $(date +%s) - START )) -ge "$TIMEOUT" ]; then
      echo "timeout reached"
      break
    fi
    echo "sleeping ${INTERVAL}s ..."
    sleep "$INTERVAL"
  done
else
  if run_once; then
    :
  else
    if [ "$REQUIRE_COMPLETE" -eq 1 ]; then
      exit 1
    fi
  fi
fi

echo ""
echo "Local results: $REPO/results/runs"
echo "Quick status:  bash scripts/pull_results.sh --status"
echo "Paper table:   MODELS=\"dcd:all_dense sacm_official:default unet:default cs2net:default maskvsc:default\" bash scripts/build_paper_table.sh"
