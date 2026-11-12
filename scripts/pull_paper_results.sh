#!/usr/bin/env bash
# One-click pull of the server-side paper experiment results.
#
# Pulls the 20-dataset eval tables for:
#   unet cs2net cwsam knowsam maskvsc nnwnet
#   sacm_official sam_med2d sam_octa segdino
# plus the special roots:
#   sacm_stc_ssh_full_ab/ssh
#   sacm_stc_fixed3_800/eval
#
# The generic scripts/pull_results.sh handles the normal
# <model>/<variant>/seed_<s>/eval/tables/all_metrics.csv layout.  fixed3 uses
# eval/seed_<s>/... and is pulled separately.
#
# Usage (local repo root):
#   bash scripts/pull_paper_results.sh
#   DRY_RUN=1 bash scripts/pull_paper_results.sh
#   REQUIRE_COMPLETE=1 bash scripts/pull_paper_results.sh
#   WITH_PREDICTIONS=1 bash scripts/pull_paper_results.sh
#
# Env overrides:
#   SERVER=zkyd@172.16.1.7
#   TARGET=/home/zkyd/code/sacm-plus
#   SEEDS="0 1 2 3"
#   EXPECTED_DATASETS=20
#   BASELINES="unet:default ... segdino:default"
#   EXTRA_MODEL=sacm_stc_ssh_full_ab:ssh
#   PULL_EXTRA=1 PULL_FIXED3=1 PULL_SUMMARY=1
#   WITH_PREDICTIONS=0   pull prediction PNGs too (very large)
#   REQUIRE_COMPLETE=0   exit 1 if any selected eval is missing/partial
#   DRY_RUN=0
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SERVER="${SERVER:-zkyd@172.16.1.7}"
TARGET="${TARGET:-/home/zkyd/code/sacm-plus}"
SEEDS_STR="${SEEDS:-0 1 2 3}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"
BASELINES="${BASELINES:-unet:default cs2net:default cwsam:default knowsam:default maskvsc:default nnwnet:default sacm_official:default sam_med2d:default sam_octa:default segdino:default}"
EXTRA_MODEL="${EXTRA_MODEL:-sacm_stc_ssh_full_ab:ssh}"
PULL_EXTRA="${PULL_EXTRA:-1}"
PULL_FIXED3="${PULL_FIXED3:-1}"
PULL_SUMMARY="${PULL_SUMMARY:-1}"
WITH_PREDICTIONS="${WITH_PREDICTIONS:-0}"
REQUIRE_COMPLETE="${REQUIRE_COMPLETE:-0}"
DRY_RUN="${DRY_RUN:-0}"

for name in PULL_EXTRA PULL_FIXED3 PULL_SUMMARY WITH_PREDICTIONS \
             REQUIRE_COMPLETE DRY_RUN; do
  value="${!name}"
  case "${value,,}" in
    1|true|yes|on) printf -v "$name" '%s' 1 ;;
    0|false|no|off) printf -v "$name" '%s' 0 ;;
    *) echo "ERROR: $name must be 0/1 (or true/false/yes/no/on/off)" >&2
       exit 2 ;;
  esac
done
if ! [[ "$EXPECTED_DATASETS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EXPECTED_DATASETS must be an integer" >&2
  exit 2
fi
read -r -a SEED_ARGS <<< "$SEEDS_STR"
if [ "${#SEED_ARGS[@]}" -eq 0 ]; then
  echo "ERROR: SEEDS is empty" >&2
  exit 2
fi
if [ ! -f scripts/pull_results.sh ]; then
  echo "ERROR: missing scripts/pull_results.sh" >&2
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
export PYTHON
export SERVER TARGET EXPECTED_DATASETS

MODEL_SPECS="$BASELINES"
if [ "$PULL_EXTRA" = 1 ] && [ -n "$EXTRA_MODEL" ]; then
  MODEL_SPECS="$MODEL_SPECS $EXTRA_MODEL"
fi
if [ -z "${MODEL_SPECS// /}" ]; then
  echo "ERROR: no baseline/model specs selected" >&2
  exit 2
fi
if [ "$WITH_PREDICTIONS" = 1 ]; then
  echo "WARNING: WITH_PREDICTIONS=1 pulls prediction PNGs and can be very large." >&2
fi

# Match pull_results.sh's ControlPath so all SSH/rsync calls share one auth.
CONTROL_DIR="${TMPDIR:-/tmp}/sacm-plus-ssh-$(id -u)"
mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"
CONTROL_PATH="$CONTROL_DIR/%C"
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$CONTROL_PATH"
          -o ControlPersist=600 -o ServerAliveInterval=30
          -o ServerAliveCountMax=4 -o ConnectTimeout=10)
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

echo "============================================================"
echo " Pull paper test results from $SERVER:$TARGET"
echo " seeds: ${SEEDS_STR}   expected datasets: ${EXPECTED_DATASETS}"
echo " models: ${MODEL_SPECS}"
echo " fixed3: ${PULL_FIXED3}  summary: ${PULL_SUMMARY}"
if [ "$DRY_RUN" = 1 ]; then
  echo " mode: DRY RUN (only show transfers)"
fi
echo "============================================================"

if ! ssh "${SSH_OPTS[@]}" "$SERVER" "test -d '$TARGET/results/runs'"; then
  echo "ERROR: cannot access $SERVER:$TARGET/results/runs" >&2
  echo "Check SSH access and TARGET. Expected layout:" >&2
  echo "  $TARGET/results/runs/<model>/default/seed_<s>/eval/tables/" >&2
  exit 3
fi

# ---------------------------------------------------------------------------
# 1. Normal <model>/<variant>/seed_<s>/eval/tables layout.
# ---------------------------------------------------------------------------
PULL_ARGS=(--models "$MODEL_SPECS" --seeds "$SEEDS_STR" --retries 3)
if [ "$WITH_PREDICTIONS" = 1 ]; then
  PULL_ARGS+=(--with-predictions)
fi
if [ "$DRY_RUN" = 1 ]; then
  PULL_ARGS+=(--dry-run)
fi
echo "[1/3] pulling baseline + SSH eval tables"
if ! env EXPECTED_DATASETS="$EXPECTED_DATASETS" \
    bash scripts/pull_results.sh "${PULL_ARGS[@]}"; then
  echo "ERROR: baseline pull failed; see the rsync/SSH message above." >&2
  exit 3
fi

# ---------------------------------------------------------------------------
# 2. fixed3 uses a special eval/seed_<s> layout, not model/variant/seed.
# ---------------------------------------------------------------------------
if [ "$PULL_FIXED3" = 1 ]; then
  remote_fixed3="$TARGET/results/runs/sacm_stc_fixed3_800"
  local_fixed3="$REPO/results/runs/sacm_stc_fixed3_800"
  if ssh "${SSH_OPTS[@]}" "$SERVER" "test -d '$remote_fixed3'"; then
    echo "[2/3] pulling fixed3 eval/seed_<s> tables"
    mkdir -p "$local_fixed3"
    fixed3_args=(-az --partial --update --prune-empty-dirs --timeout=120
                 --info=progress2 -e "$RSYNC_SSH")
    if [ "$DRY_RUN" = 1 ]; then
      fixed3_args+=(--dry-run)
    fi
    if [ "$WITH_PREDICTIONS" = 0 ]; then
      fixed3_args+=(--exclude='predictions/')
    fi
    if ! rsync "${fixed3_args[@]}" \
        "$SERVER:$remote_fixed3/" "$local_fixed3/"; then
      echo "ERROR: fixed3 rsync failed." >&2
      exit 3
    fi
  else
    echo "WARNING: server fixed3 root is missing: $remote_fixed3" >&2
  fi
else
  echo "[2/3] fixed3 pull disabled (PULL_FIXED3=0)"
fi

# ---------------------------------------------------------------------------
# 3. Optional fixed3 vs official summary (small, read-only analysis).
# ---------------------------------------------------------------------------
if [ "$PULL_SUMMARY" = 1 ]; then
  remote_summary="$TARGET/results/ablations/sacm_stc_fixed3_800"
  local_summary="$REPO/results/ablations/sacm_stc_fixed3_800"
  if ssh "${SSH_OPTS[@]}" "$SERVER" "test -d '$remote_summary'"; then
    echo "[3/3] pulling fixed3 summary"
    mkdir -p "$local_summary"
    summary_args=(-az --partial --update --prune-empty-dirs --timeout=120
                  --info=progress2 -e "$RSYNC_SSH")
    if [ "$DRY_RUN" = 1 ]; then
      summary_args+=(--dry-run)
    fi
    if ! rsync "${summary_args[@]}" \
        "$SERVER:$remote_summary/" "$local_summary/"; then
      echo "ERROR: fixed3 summary rsync failed." >&2
      exit 3
    fi
  else
    echo "WARNING: server fixed3 summary is missing: $remote_summary" >&2
  fi
else
  echo "[3/3] summary pull disabled (PULL_SUMMARY=0)"
fi

# ---------------------------------------------------------------------------
# 4. Local coverage check: every selected run must have 20 dataset rows.
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
  echo "dry-run: skipping local coverage check"
  exit 0
fi

echo "[check] local 20-dataset coverage"
set +e
"$PYTHON" - "$REPO" "$EXPECTED_DATASETS" "$SEEDS_STR" \
  "$MODEL_SPECS" "$PULL_FIXED3" <<'PY'
import csv
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
expected_n = int(sys.argv[2])
seeds = sys.argv[3].split()
specs = sys.argv[4].split()
pull_fixed3 = sys.argv[5] == '1'

if pull_fixed3:
    specs = specs + ['sacm_stc_fixed3_800:fixed3']


def rows_in(path):
    if not path.is_file():
        return None
    with path.open(newline='') as handle:
        return [r for r in csv.DictReader(handle) if r.get('dataset')]


ok_all = True
for spec in specs:
    if ':' in spec:
        model, variant = spec.split(':', 1)
    else:
        model, variant = spec, 'default'
    for seed in seeds:
        if model == 'sacm_stc_fixed3_800':
            csv_path = (repo / 'results' / 'runs' / model / 'eval'
                        / f'seed_{seed}' / 'tables' / 'all_metrics.csv')
            pred_dir = (repo / 'results' / 'runs' / model
                        / 'predictions' / f'seed_{seed}')
        else:
            csv_path = (repo / 'results' / 'runs' / model / variant
                        / f'seed_{seed}' / 'eval' / 'tables' / 'all_metrics.csv')
            pred_dir = (repo / 'results' / 'runs' / model / variant
                        / f'seed_{seed}' / 'predictions')
        rows = rows_in(csv_path)
        label = f'{model}/{variant} seed_{seed}'
        if rows is None:
            extra = ''
            if pred_dir.is_dir():
                extra = ' (predictions present; metrics missing)'
            print(f'  {label}: MISSING{extra}')
            ok_all = False
            continue
        n = len(rows)
        state = 'OK' if n >= expected_n else 'PARTIAL'
        if n < expected_n:
            ok_all = False
        print(f'  {label}: {n}/{expected_n} {state}')
        if n == 0:
            print(f'    {csv_path}')
print('coverage:', 'COMPLETE' if ok_all else 'INCOMPLETE')
raise SystemExit(0 if ok_all else 1)
PY
CHECK_RC=$?
set -e
if [ "$CHECK_RC" -ne 0 ]; then
  if [ "$REQUIRE_COMPLETE" = 1 ]; then
    echo "ERROR: selected results are incomplete (REQUIRE_COMPLETE=1)." >&2
    exit 1
  fi
  echo "WARNING: selected results are incomplete; partial data was pulled." >&2
fi

echo "============================================================"
echo " Done."
echo " local runs:      $REPO/results/runs"
echo " fixed3 summary:  $REPO/results/ablations/sacm_stc_fixed3_800"
echo " fixed3 eval:     $REPO/results/runs/sacm_stc_fixed3_800/eval/seed_*/tables/"
echo "============================================================"
