#!/usr/bin/env bash
# One-click pull of the fixed3 lean-ablation results.
#
# Pulls, by default:
#   results/runs/sacm_stc_fixed3_lean/       (eval tables + run metadata)
#   results/runs/sacm_stc_fixed3_800/eval/   (existing fixed3 4-seed eval)
#   results/ablations/sacm_stc_fixed3_lean/  (summary.csv/md/html/json)
#   results/logs/sacm_stc_fixed3_lean_*/     (driver/eval/derive logs)
#
# Predictions and checkpoints are excluded by default.  The script never uses
# rsync --delete and never overwrites a newer local file.
#
# Usage:
#   bash scripts/pull_fixed3_lean_results.sh
#   DRY_RUN=1 bash scripts/pull_fixed3_lean_results.sh
#   REQUIRE_COMPLETE=0 bash scripts/pull_fixed3_lean_results.sh
#   WITH_LOGS=0 bash scripts/pull_fixed3_lean_results.sh
#   WITH_PREDICTIONS=1 bash scripts/pull_fixed3_lean_results.sh
#   WITH_CHECKPOINTS=1 bash scripts/pull_fixed3_lean_results.sh
#
# Env overrides:
#   SERVER=zkyd@172.16.1.7
#   TARGET=/home/zkyd/code/sacm-plus
#   SEEDS="0 1 2 3"
#   EXPECTED_DATASETS=20
#   LEAN_ROOT=results/runs/sacm_stc_fixed3_lean
#   FIXED3_ROOT=results/runs/sacm_stc_fixed3_800/eval
#   SUMMARY_ROOT=results/ablations/sacm_stc_fixed3_lean
#   LOG_PREFIX=sacm_stc_fixed3_lean_
#   WITH_LOGS=1 WITH_PREDICTIONS=0 WITH_CHECKPOINTS=0
#   REQUIRE_COMPLETE=1
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SERVER="${SERVER:-zkyd@172.16.1.7}"
TARGET="${TARGET:-/home/zkyd/code/sacm-plus}"
SEEDS_STR="${SEEDS:-0 1 2 3}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"
LEAN_ROOT="${LEAN_ROOT:-results/runs/sacm_stc_fixed3_lean}"
FIXED3_ROOT="${FIXED3_ROOT:-results/runs/sacm_stc_fixed3_800/eval}"
SUMMARY_ROOT="${SUMMARY_ROOT:-results/ablations/sacm_stc_fixed3_lean}"
LOG_PREFIX="${LOG_PREFIX:-sacm_stc_fixed3_lean_}"
WITH_LOGS="${WITH_LOGS:-1}"
WITH_PREDICTIONS="${WITH_PREDICTIONS:-0}"
WITH_CHECKPOINTS="${WITH_CHECKPOINTS:-0}"
REQUIRE_COMPLETE="${REQUIRE_COMPLETE:-1}"
DRY_RUN="${DRY_RUN:-0}"
RETRIES="${RETRIES:-3}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/pull_fixed3_lean_results.sh [options]

Options:
  --dry-run               show rsync transfers; do not write
  --seeds "0 1 2 3"       seed list (default: 0 1 2 3)
  --with-logs             pull results/logs/sacm_stc_fixed3_lean_* (default)
  --no-logs               skip logs
  --with-predictions      also pull predictions/ (large)
  --with-checkpoints      also pull model weights (large)
  --require-complete      exit 1 unless every seed has 20-dataset eval (default)
  --no-require-complete   report missing tables but exit 0
  --retries N             rsync retry attempts (default: 3)
  -h, --help              show this help

Env: SERVER, TARGET, EXPECTED_DATASETS, LEAN_ROOT, FIXED3_ROOT,
     SUMMARY_ROOT, LOG_PREFIX, WITH_LOGS, WITH_PREDICTIONS,
     WITH_CHECKPOINTS, REQUIRE_COMPLETE, DRY_RUN, RETRIES
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --seeds)
      [ "$#" -ge 2 ] || { echo "missing value for --seeds" >&2; exit 2; }
      shift; SEEDS_STR="$1"
      ;;
    --with-logs) WITH_LOGS=1 ;;
    --no-logs) WITH_LOGS=0 ;;
    --with-predictions) WITH_PREDICTIONS=1 ;;
    --no-predictions) WITH_PREDICTIONS=0 ;;
    --with-checkpoints) WITH_CHECKPOINTS=1 ;;
    --no-checkpoints) WITH_CHECKPOINTS=0 ;;
    --require-complete) REQUIRE_COMPLETE=1 ;;
    --no-require-complete) REQUIRE_COMPLETE=0 ;;
    --retries)
      [ "$#" -ge 2 ] || { echo "missing value for --retries" >&2; exit 2; }
      shift; RETRIES="$1"
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

as_bool() {
  case "${1,,}" in
    1|true|yes|on) printf '1\n' ;;
    0|false|no|off) printf '0\n' ;;
    *) echo "ERROR: expected 0/1, got '$1'" >&2; exit 2 ;;
  esac
}
WITH_LOGS=$(as_bool "$WITH_LOGS")
WITH_PREDICTIONS=$(as_bool "$WITH_PREDICTIONS")
WITH_CHECKPOINTS=$(as_bool "$WITH_CHECKPOINTS")
REQUIRE_COMPLETE=$(as_bool "$REQUIRE_COMPLETE")
DRY_RUN=$(as_bool "$DRY_RUN")

read -r -a SEED_ARGS <<< "${SEEDS_STR//,/ }"
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
if ! [[ "$EXPECTED_DATASETS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EXPECTED_DATASETS must be positive" >&2
  exit 2
fi
if ! [[ "$RETRIES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: RETRIES must be positive" >&2
  exit 2
fi
NORMALIZED_SEEDS="${SEED_ARGS[*]}"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  elif [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python3
  fi
fi
export PYTHON

CONTROL_DIR="${TMPDIR:-/tmp}/sacm-plus-ssh-$(id -u)"
mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$CONTROL_DIR/%C"
          -o ControlPersist=600 -o ServerAliveInterval=30
          -o ServerAliveCountMax=4 -o ConnectTimeout=10)
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

remote_exists() {
  local path="$1"
  case "$path" in
    /*) ;;
    *) path="$TARGET/$path" ;;
  esac
  ssh "${SSH_OPTS[@]}" "$SERVER" "test -e '$path'" >/dev/null 2>&1
}

RSYNC_ARGS=(-az --partial --update --prune-empty-dirs --timeout=120
            --info=progress2 -e "$RSYNC_SSH")
if [ "$WITH_PREDICTIONS" -eq 0 ]; then
  RSYNC_ARGS+=(--exclude='predictions*/' --exclude='visualizations/')
fi
if [ "$WITH_CHECKPOINTS" -eq 0 ]; then
  RSYNC_ARGS+=(--exclude='*.pth' --exclude='*.pt' --exclude='*.safetensors')
fi
RSYNC_ARGS+=(--exclude='checkpoints/' --exclude='wandb/'
             --exclude='.resume_state*' --exclude='.part')
if [ "$DRY_RUN" -eq 1 ]; then
  RSYNC_ARGS+=(--dry-run)
fi

pull_dir() {
  local rel="$1"
  local remote="$TARGET/$rel"
  local local_path="$REPO/$rel"
  local attempt=1
  echo "[rsync] $SERVER:$remote/ -> $local_path/"
  if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$local_path"
  fi
  while :; do
    if rsync "${RSYNC_ARGS[@]}" "$SERVER:$remote/" "$local_path/"; then
      return 0
    fi
    if [ "$attempt" -ge "$RETRIES" ]; then
      echo "ERROR: rsync failed after $RETRIES attempt(s): $rel" >&2
      return 1
    fi
    attempt=$((attempt + 1))
    echo "rsync failed; retry $attempt/$RETRIES in 10s ..." >&2
    sleep 10
  done
}

echo "============================================================"
echo " Pull fixed3 lean-ablation results"
echo " server:  $SERVER:$TARGET"
echo " seeds:   $NORMALIZED_SEEDS"
echo " lean:    $LEAN_ROOT"
echo " fixed3:  $FIXED3_ROOT"
echo " summary: $SUMMARY_ROOT"
echo " logs:    $WITH_LOGS  predictions: $WITH_PREDICTIONS  checkpoints: $WITH_CHECKPOINTS"
if [ "$DRY_RUN" -eq 1 ]; then
  echo " mode:    DRY RUN"
fi
echo "============================================================"

echo "[ssh] checking $SERVER ..."
if ! ssh "${SSH_OPTS[@]}" "$SERVER" true; then
  echo "ERROR: SSH authentication failed for $SERVER." >&2
  echo "Check: ssh $SERVER 'test -d $TARGET'" >&2
  exit 3
fi

if ! remote_exists "$LEAN_ROOT"; then
  echo "ERROR: remote lean root missing: $TARGET/$LEAN_ROOT" >&2
  echo "Check that the one-click ablation completed on the server." >&2
  exit 3
fi
if ! remote_exists "$FIXED3_ROOT"; then
  echo "ERROR: remote fixed3 eval root missing: $TARGET/$FIXED3_ROOT" >&2
  exit 3
fi

if [ "$REQUIRE_COMPLETE" -eq 1 ]; then
  missing=()
  for seed in "${SEED_ARGS[@]}"; do
    for sub in \
      "$LEAN_ROOT/gating_fusion/seed_${seed}/eval" \
      "$LEAN_ROOT/nofusion/seed_${seed}/eval_fixed3" \
      "$LEAN_ROOT/native_control/seed_${seed}/eval" \
      "$LEAN_ROOT/notopo/seed_${seed}/eval_fixed3"; do
      path="$sub/tables/all_metrics.csv"
      remote_exists "$path" || missing+=("$path")
    done
    path="$FIXED3_ROOT/seed_${seed}/tables/all_metrics.csv"
    remote_exists "$path" || missing+=("$path")
  done
  for path in "$SUMMARY_ROOT/summary.md" "$SUMMARY_ROOT/summary.csv"; do
    remote_exists "$path" || missing+=("$path")
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: remote ablation is incomplete: ${#missing[@]} expected file(s) missing." >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo "The eval/summary may still be running. Re-run later, or use" >&2
    echo "--no-require-complete to pull partial data." >&2
    exit 3
  fi
fi

pull_dir "$LEAN_ROOT"
pull_dir "$FIXED3_ROOT"

if remote_exists "$SUMMARY_ROOT"; then
  pull_dir "$SUMMARY_ROOT"
else
  echo "WARNING: remote summary root missing: $TARGET/$SUMMARY_ROOT" >&2
fi

if [ "$WITH_LOGS" -eq 1 ] && remote_exists "results/logs"; then
  mapfile -t LOG_NAMES < <(
    ssh "${SSH_OPTS[@]}" "$SERVER" \
      "find '$TARGET/results/logs' -mindepth 1 -maxdepth 1 -type d \
       -name '${LOG_PREFIX}*' -printf '%f\n' 2>/dev/null | sort"
  )
  for name in "${LOG_NAMES[@]}"; do
    pull_dir "results/logs/$name"
  done
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "dry-run: skipping local coverage check"
  exit 0
fi

echo "[check] local ablation coverage"
"$PYTHON" - "$REPO" "$LEAN_ROOT" "$FIXED3_ROOT" \
  "$SUMMARY_ROOT" "$NORMALIZED_SEEDS" "$EXPECTED_DATASETS" \
  "$REQUIRE_COMPLETE" <<'PY'
import csv
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
lean_root = repo / sys.argv[2]
fixed3_root = repo / sys.argv[3]
summary_root = repo / sys.argv[4]
seeds = [int(x) for x in sys.argv[5].split()]
expected = int(sys.argv[6])
require = sys.argv[7] == '1'

# arm -> relative eval root under lean_root
arms = {
    'gating_fusion': Path('gating_fusion') / 'seed_{seed}' / 'eval',
    'nofusion': Path('nofusion') / 'seed_{seed}' / 'eval_fixed3',
    'native_control': Path('native_control') / 'seed_{seed}' / 'eval',
    'notopo': Path('notopo') / 'seed_{seed}' / 'eval_fixed3',
}

def dataset_rows(table: Path):
    if not table.is_file():
        return 'missing all_metrics.csv'
    try:
        with table.open(newline='') as handle:
            rows = [r for r in csv.DictReader(handle) if r.get('dataset')]
    except (OSError, csv.Error) as exc:
        return f'unreadable: {exc}'
    names = {r['dataset'] for r in rows}
    if len(names) < expected:
        return f'only {len(names)}/{expected} datasets'
    for row in rows:
        for metric in ('dice_mean', 'iou_mean', 'cldice_mean', 'hd95_mean'):
            if row.get(metric) in (None, ''):
                return f'missing {metric}'
    return 'OK'

ok = True
for seed in seeds:
    fixed_table = fixed3_root / f'seed_{seed}' / 'tables' / 'all_metrics.csv'
    state = dataset_rows(fixed_table)
    print(f'  fixed3/seed_{seed}: {state}')
    ok = ok and state == 'OK'
    for arm, template in arms.items():
        table = lean_root / template.as_posix().format(seed=seed) / \
            'tables/all_metrics.csv'
        state = dataset_rows(table)
        print(f'  {arm}/seed_{seed}: {state}')
        ok = ok and state == 'OK'

for name in ('summary.md', 'summary.csv', 'summary.html'):
    path = summary_root / name
    state = 'OK' if path.is_file() else 'MISSING'
    print(f'  {name}: {state}')
    if name in ('summary.md', 'summary.csv'):
        ok = ok and path.is_file()

official = repo / 'results/runs/sacm_official/default'
missing_official = [
    str(official / f'seed_{seed}' / 'eval/tables/all_metrics.csv')
    for seed in seeds
    if not (official / f'seed_{seed}' / 'eval/tables/all_metrics.csv').is_file()
]
if missing_official:
    print('  official reference: MISSING (needed for the full summary table)')
    for path in missing_official[:2]:
        print(f'    {path}')
else:
    print('  official reference: OK')

print('coverage:', 'COMPLETE' if ok else 'INCOMPLETE')
if require and not ok:
    raise SystemExit(1)
PY

echo "============================================================"
echo " Done."
echo " lean eval:   $LEAN_ROOT"
echo " fixed3 eval: $FIXED3_ROOT"
echo " summary:     $SUMMARY_ROOT/summary.md"
echo "============================================================"
