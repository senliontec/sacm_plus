#!/usr/bin/env bash
# One-click switch from the old six-arm DCD eval tables to best-only
# results.
#
# What it does:
#   1. archives the six legacy DCD eval directories and the old
#      scope/structure analysis directory;
#   2. preserves all training directories and best checkpoints;
#   3. removes temporary resume/periodic/backup checkpoints from results/runs;
#   4. verifies every remaining eval manifest points at a best checkpoint;
#   5. optionally builds the best-only paper table.
#
# It is idempotent: already archived eval dirs are skipped.
#
# Usage (local repo root, and separately on the server):
#   bash scripts/switch_to_best_only.sh
#   bash scripts/switch_to_best_only.sh --dry-run
#   bash scripts/switch_to_best_only.sh --paper-table
#
# The server repo must have this script synced.  Run it there first, then run
# it locally; otherwise a later pull can restore the legacy eval directories.
#
# Env overrides:
#   MAIN_VARIANT=das_as_dep
#   PAPER_MODELS="dcd:das_as_dep sacm_official:default ..."
#   PAPER_DATASETS="CHASEDB1 ... WIRE"
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

DRY_RUN=0
PAPER_TABLE=0
CLEAN_TEMP=1
FORCE=0
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="results/archive/dcd_old_six_eval_${STAMP}"
LOG_ROOT="results/logs"
LOCK_FILE="$LOG_ROOT/switch_to_best_only.lock"
MAIN_VARIANT="${MAIN_VARIANT:-das_as_dep}"
PAPER_MODELS="${PAPER_MODELS:-dcd:${MAIN_VARIANT} sacm_official:default unet:default cs2net:default maskvsc:default}"
PAPER_DATASETS="${PAPER_DATASETS:-CHASEDB1 CrackTree200 CREMI DCA1 DRIVE CHCrack5k FIVES HRF LES-AV OCTA-500 DeepGlobe ROAD TopoMortar TYRE WIRE}"

OLD_ARMS=(
  b0_selected_base
  b1_all_dense_base
  ss_selected_dense_selected_structure
  sa_selected_dense_all_structure
  b2_all_dense_selected_structure
  b3_all_dense_all_structure
)

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --paper-table) PAPER_TABLE=1 ;;
    --no-clean-temp) CLEAN_TEMP=0 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ ! -d results/runs ]; then
  echo "ERROR: run this from the repo root (results/runs not found)." >&2
  exit 2
fi

if [ "$FORCE" -ne 1 ] && command -v pgrep >/dev/null 2>&1; then
  # Check actual training/eval workers, not stale outer wrapper shells.
  # A leftover wrapper with no child must not block a data-only archive
  # operation.
  if pgrep -af 'baseline_zoo\.run_one|run_baseline_test\.py|run_dcd_ablation\.py' \
      >/tmp/switch_best_only_active.$$ 2>/dev/null; then
    if [ -s "/tmp/switch_best_only_active.$$" ]; then
      echo "ERROR: training/eval processes are still running:" >&2
      cat "/tmp/switch_best_only_active.$$" >&2
      rm -f "/tmp/switch_best_only_active.$$"
      echo "Wait for them to finish, or pass --force." >&2
      exit 2
    fi
  fi
  rm -f "/tmp/switch_best_only_active.$$"
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

if [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$LOG_ROOT"
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "ERROR: another switch_to_best_only run is active." >&2
    exit 2
  fi
fi

move_path() {
  local src="$1"
  local dst="$2"
  if [ ! -e "$src" ]; then
    return 1
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "would move: $src -> $dst"
  else
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
    echo "moved: $src -> $dst"
  fi
  return 0
}

echo "=== switch_to_best_only ==="
echo "repo:   $REPO"
echo "archive: $ARCHIVE"
echo "dry-run: $DRY_RUN"
echo "paper-table: $PAPER_TABLE"

if [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$ARCHIVE"
else
  echo "dry-run: archive directory will be created only on execution"
fi

MOVED_EVAL=0
for arm in "${OLD_ARMS[@]}"; do
  for seed in 0 1 2 3; do
    src="results/runs/dcd/${arm}/seed_${seed}/eval"
    dst="${ARCHIVE}/runs/${arm}/seed_${seed}/eval"
    if move_path "$src" "$dst"; then
      MOVED_EVAL=$((MOVED_EVAL + 1))
    fi
  done
done
echo "legacy eval directories moved: $MOVED_EVAL"

if move_path "results/ablations/dcd" "${ARCHIVE}/ablations/dcd"; then
  :
else
  echo "old scope/structure analysis already absent"
fi

if [ "$DRY_RUN" -eq 0 ]; then
  cat > "${ARCHIVE}/README.md" <<EOF
# Archived legacy six-arm eval

These eval artifacts were produced by the old six-arm DCD scope/structure
matrix and are not best-only results.  They are retained for audit only.

Archived at: ${STAMP}
Source repo: ${REPO}
EOF
fi

if [ "$CLEAN_TEMP" -eq 1 ]; then
  echo "cleaning temporary checkpoints under results/runs"
  if [ "$DRY_RUN" -eq 1 ]; then
    find results/runs -type f \
      \( -name '*last*.pth' -o -name '.resume_state.pth' \
         -o -name 'best_model.before_*.pth' \) -print
    find results/runs -type d -name checkpoints -print
  else
    find results/runs -type f \
      \( -name '*last*.pth' -o -name '.resume_state.pth' \
         -o -name 'best_model.before_*.pth' \) -delete
    find results/runs -type d -name checkpoints -prune -exec rm -rf {} +
  fi
fi

echo "verifying remaining eval manifests..."
if [ "$DRY_RUN" -eq 1 ]; then
  echo "dry-run: verification skipped because legacy eval directories were not moved"
  VERIFY_STATUS=0
else
  if "$PYTHON" - "$REPO" <<'PY'
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
root = repo / 'results' / 'runs'
failures = []
checked = 0
for manifest in sorted(root.glob('*/*/seed_*/eval/run_manifest.json')):
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f'{manifest}: unreadable ({exc})')
        continue
    if payload.get('status') not in (None, 'done'):
        failures.append(f'{manifest}: status={payload.get("status")!r}')
        continue
    native = str(payload.get('native_checkpoint') or '')
    name = Path(native).name.lower()
    kind = str(payload.get('checkpoint_kind') or '')
    checked += 1
    if ('last' in name or 'resume' in name or 'before' in name):
        failures.append(f'{manifest}: non-best checkpoint {native!r}')
    elif kind != 'directory' and 'best' not in name:
        failures.append(f'{manifest}: unexpected checkpoint {native!r}')

print(f'eval manifests checked: {checked}')
if failures:
    print('FAILED:')
    for item in failures:
        print('  ', item)
    raise SystemExit(1)
print('OK: all remaining eval manifests use best checkpoints')
PY
  then
    VERIFY_STATUS=0
  else
    VERIFY_STATUS=1
  fi
fi

PAPER_STATUS=0
if [ "$PAPER_TABLE" -eq 1 ]; then
  echo "building best-only paper table (main variant: $MAIN_VARIANT)"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "would run: MODELS=\"$PAPER_MODELS\" DATASETS=\"$PAPER_DATASETS\" bash scripts/build_paper_table.sh"
  else
    MODELS="$PAPER_MODELS" DATASETS="$PAPER_DATASETS" \
      bash scripts/build_paper_table.sh || PAPER_STATUS=$?
  fi
fi

echo "============================================================"
echo " switch_to_best_only finished"
echo " archive: $ARCHIVE"
echo " best-only reports:"
echo "   results/ablations/dcd_das/analysis/ablation_report.md"
echo "   results/ablations/dcd_native_control/analysis/ablation_report.md"
echo " verify status: $VERIFY_STATUS"
echo " paper status: $PAPER_STATUS"
echo "============================================================"
if [ "$VERIFY_STATUS" -ne 0 ] || [ "$PAPER_STATUS" -ne 0 ]; then
  exit 1
fi
exit 0
