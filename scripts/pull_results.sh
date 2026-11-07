#!/usr/bin/env bash
# Incrementally pull server-side results/runs into the local repo.
#
# Default pull set (small analysis artifacts, no weights / predictions):
#   eval/**, metrics_history.json, checkpoint.json, train_config.json,
#   run_info.json, wandb_run.json, val_metrics_by_dataset.csv,
#   provenance/**, train.log, eval.log
#
# The script never uses rsync --delete and never overwrites newer local files.
# It is safe to run repeatedly while the server is still evaluating.
#
# Usage:
#   bash scripts/pull_results.sh
#   bash scripts/pull_results.sh --models "sacm_stc:all_dense sacm_official:default" --seeds "0 1"
#   bash scripts/pull_results.sh --watch 300
#   bash scripts/pull_results.sh --until-complete --interval 300
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

DEFAULT_MODELS=(sacm_stc:all_dense sacm_official:default unet:default cs2net:default maskvsc:default)
DEFAULT_SEEDS=(0 1 2 3)

MODEL_FILTERS=()
SEED_FILTERS=()
DATASET_FILTERS=()
WITH_PREDICTIONS=0
WITH_CHECKPOINTS=0
DRY_RUN=0
REQUIRE_COMPLETE=0
WATCH=0
INTERVAL=300
TIMEOUT=0

usage() {
  cat <<'EOF'
Pull server-side results/runs into the local repo (incremental, no delete).

Usage:
  bash scripts/pull_results.sh [options]

Options:
  --models LIST        model or model:variant list; default: all models
                       e.g. "sacm_stc:all_dense sacm_official:default"
  --seeds LIST         seed list; default: all seeds
  --datasets LIST      only with --with-predictions; dataset filter
  --with-predictions   also pull predictions/ (can be very large)
  --with-checkpoints   also pull best_model.pth / stage*_best.pth
  --dry-run            show what rsync would transfer; do not write
  --watch [SECONDS]    pull repeatedly (default interval 300s)
  --until-complete     repeat until selected runs have 20 dataset rows
  --interval SECONDS   watch interval (default 300)
  --timeout SECONDS    stop watch after this many seconds (0 = never)
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
RAW="$TMP/raw.txt"
LIST="$TMP/files.txt"
EXPECTED="$TMP/expected.txt"

CONTROL_DIR="${TMPDIR:-/tmp}/sacm-plus-ssh-$(id -u)"
mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"
CONTROL_PATH="$CONTROL_DIR/%C"
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$CONTROL_PATH"
          -o ControlPersist=600)
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

if ! ssh "${SSH_OPTS[@]}" "$SERVER" "test -d '$TARGET/results/runs'" >/dev/null 2>&1; then
  echo "ERROR: cannot access $SERVER:$TARGET/results/runs" >&2
  echo "Check SERVER/TARGET and network/SSH access." >&2
  exit 3
fi

find_expression() {
  local expr="-path '*/eval/*'"
  expr+=" -o -name 'metrics_history.json'"
  expr+=" -o -name 'checkpoint.json'"
  expr+=" -o -name 'train_config.json'"
  expr+=" -o -name 'run_info.json'"
  expr+=" -o -name 'wandb_run.json'"
  expr+=" -o -name 'val_metrics_by_dataset.csv'"
  expr+=" -o -path '*/provenance/*'"
  expr+=" -o -name 'train.log'"
  expr+=" -o -name 'eval.log'"
  if [ "$WITH_PREDICTIONS" -eq 1 ]; then
    expr+=" -o -path '*/predictions/*'"
  fi
  if [ "$WITH_CHECKPOINTS" -eq 1 ]; then
    expr+=" -o -name 'best_model.pth'"
    expr+=" -o -name 'stage1_best.pth'"
    expr+=" -o -name 'stage2_best.pth'"
    expr+=" -o -name 'last_model.pth'"
  fi
  printf '%s' "$expr"
}

filter_files() {
  local model_filter=1
  local seed_filter=1
  if [ ${#MODEL_FILTERS[@]} -eq 0 ] || [ "${MODEL_FILTERS[0]}" = "all" ]; then
    model_filter=0
  fi
  if [ ${#SEED_FILTERS[@]} -eq 0 ] || [ "${SEED_FILTERS[0]}" = "all" ]; then
    seed_filter=0
  fi
  : > "$LIST"
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    if [ "$model_filter" -eq 1 ]; then
      local ok=0
      local m prefix
      for m in "${MODEL_FILTERS[@]}"; do
        if [[ "$m" == *:* ]]; then
          prefix="results/runs/${m%%:*}/${m#*:}/"
        else
          prefix="results/runs/$m/"
        fi
        if [[ "$f" == "$prefix"* ]]; then ok=1; break; fi
      done
      [ "$ok" -eq 1 ] || continue
    fi
    if [ "$seed_filter" -eq 1 ]; then
      local ok=0
      local s
      for s in "${SEED_FILTERS[@]}"; do
        if [[ "$f" == */"seed_$s/"* ]]; then ok=1; break; fi
      done
      [ "$ok" -eq 1 ] || continue
    fi
    if [ ${#DATASET_FILTERS[@]} -gt 0 ] && [[ "$f" == */predictions/* ]]; then
      local rest="${f#*/predictions/}"
      rest="${rest#*/}"
      local ds="${rest%%/*}"
      local ok=0
      local d
      for d in "${DATASET_FILTERS[@]}"; do
        if [ "$ds" = "$d" ]; then ok=1; break; fi
      done
      [ "$ok" -eq 1 ] || continue
    fi
    printf '%s\n' "$f" >> "$LIST"
  done < "$RAW"
  sort -u -o "$LIST" "$LIST"
}

pull_files() {
  local expr
  expr="$(find_expression)"
  echo "=== $(date '+%F %T') querying $SERVER:$TARGET/results/runs ==="
  if ! ssh "${SSH_OPTS[@]}" "$SERVER" "cd '$TARGET' && find results/runs -type f \\( $expr \\)" > "$RAW"; then
    echo "ERROR: remote find failed" >&2
    return 1
  fi
  filter_files
  local count
  count="$(wc -l < "$LIST" | tr -d ' ')"
  echo "selected files: $count"
  if [ "$count" -eq 0 ]; then
    echo "nothing to transfer"
    return 0
  fi
  local rsync_args=(-e "$RSYNC_SSH" -avz --partial --update
                    --info=progress2 --files-from="$LIST" --relative)
  if [ "$DRY_RUN" -eq 1 ]; then
    rsync_args+=(--dry-run)
  fi
  if ! rsync "${rsync_args[@]}" "$SERVER:$TARGET/" "$REPO/"; then
    echo "ERROR: rsync failed" >&2
    return 1
  fi
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
      if [ "$model" = "sacm_stc" ]; then
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
import sys
from pathlib import Path

repo = Path(sys.argv[1])
expected_n = int(sys.argv[2])
expected_path = Path(sys.argv[3])
expected = [line.strip().split('|')
            for line in expected_path.read_text().splitlines()
            if line.strip()]

print('')
print(f'Coverage (eval/tables/all_metrics.csv; expected {expected_n} rows/run):')
ok_all = True
for model, variant, seed in expected:
    path = (repo / 'results' / 'runs' / model / variant /
            f'seed_{seed}' / 'eval' / 'tables' / 'all_metrics.csv')
    if not path.is_file():
        print(f'  {model}/{variant} seed_{seed}: MISSING')
        ok_all = False
        continue
    with path.open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row.get('dataset')]
    n = len(rows)
    status = 'OK' if n >= expected_n else 'PARTIAL'
    if n < expected_n:
        ok_all = False
    print(f'  {model}/{variant} seed_{seed}: {n}/{expected_n} {status}')
print('coverage:', 'COMPLETE' if ok_all else 'INCOMPLETE')
raise SystemExit(0 if ok_all else 1)
PY
}

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
echo "Paper table:   MODELS=\"sacm_stc:all_dense sacm_official:default unet:default cs2net:default maskvsc:default\" bash scripts/build_paper_table.sh"
