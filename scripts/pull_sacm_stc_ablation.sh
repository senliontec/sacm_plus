#!/usr/bin/env bash
# One-click pull of the completed SACM-STC ablation arms.
#
# It probes the server first and only pulls arms whose seeds each have a
# complete 20-dataset eval/tables/all_metrics.csv.  This avoids treating a
# still-training arm (currently b3/Aa) as a finished result.
#
# The actual transfer is delegated to scripts/pull_results.sh, so it uses
# the same SSH/rsync retry and include/exclude rules (no predictions or
# model weights by default).
#
# Usage (local repo root):
#   bash scripts/pull_sacm_stc_ablation.sh
#   bash scripts/pull_sacm_stc_ablation.sh --analyze
#   bash scripts/pull_sacm_stc_ablation.sh --dry-run
#   bash scripts/pull_sacm_stc_ablation.sh --seeds "0 1 2 3"
#   bash scripts/pull_sacm_stc_ablation.sh --arms "b0_selected_base ss_selected_dense_selected_structure"
#
# Env overrides:
#   SERVER=zkyd@172.16.1.7
#   TARGET=/home/zkyd/code/sacm-plus
#   EXPECTED_DATASETS=20
#
# After the pull, --analyze runs:
#   bash scripts/run_sacm_stc_ablation.sh --stage analyze \
#     --arms <completed arms> --seeds <seeds>
# and writes results/ablations/sacm_stc/analysis/ablation_report.md.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SERVER="${SERVER:-zkyd@172.16.1.7}"
TARGET="${TARGET:-/home/zkyd/code/sacm-plus}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"

DEFAULT_ARMS=(
  b0_selected_base
  b1_all_dense_base
  b2_all_dense_selected_structure
  sa_selected_dense_all_structure
  ss_selected_dense_selected_structure
  b3_all_dense_all_structure
)
ARMS=("${DEFAULT_ARMS[@]}")
SEEDS=(0 1 2 3)
ANALYZE=0
DRY_RUN=0
NO_PROBE=0

usage() {
  sed -n '2,27p' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --arms)
      [ "$#" -ge 2 ] || { echo "missing value for --arms" >&2; exit 2; }
      shift
      IFS=', ' read -r -a ARMS <<< "$1"
      ;;
    --seeds)
      [ "$#" -ge 2 ] || { echo "missing value for --seeds" >&2; exit 2; }
      shift
      IFS=', ' read -r -a SEEDS <<< "$1"
      ;;
    --analyze) ANALYZE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --no-probe) NO_PROBE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "${#ARMS[@]}" -eq 0 ]; then
  echo "ERROR: --arms is empty" >&2
  exit 2
fi
if [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "ERROR: --seeds is empty" >&2
  exit 2
fi
if ! [[ "$EXPECTED_DATASETS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EXPECTED_DATASETS must be an integer" >&2
  exit 2
fi

contains_arm() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [ "$item" = "$needle" ] && return 0
  done
  return 1
}

# Reuse pull_results.sh's SSH ControlPath so the probe and rsync share one
# authenticated connection when the server uses password auth.
CONTROL_DIR="${TMPDIR:-/tmp}/sacm-plus-ssh-$(id -u)"
mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"
SSH_OPTS=(
  -o ControlMaster=auto
  -o ControlPath="$CONTROL_DIR/%C"
  -o ControlPersist=600
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -o ConnectTimeout=10
)

COMPLETED=()
if [ "$NO_PROBE" -eq 1 ]; then
  echo "Skipping server probe (--no-probe); pulling requested arms as-is."
  COMPLETED=("${ARMS[@]}")
else
  echo "Checking SSH/path: $SERVER:$TARGET/results/runs"
  if ! ssh "${SSH_OPTS[@]}" "$SERVER" \
      "test -d '$TARGET/results/runs'"; then
    echo "ERROR: SSH authentication or target path failed." >&2
    echo "Check: ssh $SERVER 'ls $TARGET/results/runs'" >&2
    echo "Or pull explicit completed arms with --no-probe --arms ..." >&2
    exit 3
  fi

  echo "Probing completed arms on $SERVER:$TARGET ..."
  # Do not pass these as positional args to `ssh ... bash -s -- ...`:
  # SSH joins remote command words with spaces and the remote shell re-splits
  # them, so "arm1 arm2" becomes two positional args.  Pass shell-quoted
  # environment assignments instead.
  REMOTE_CMD="TARGET=$(printf '%q' "$TARGET")"
  REMOTE_CMD+=" EXPECTED=$(printf '%q' "$EXPECTED_DATASETS")"
  REMOTE_CMD+=" ARMS=$(printf '%q' "${ARMS[*]}")"
  REMOTE_CMD+=" SEEDS=$(printf '%q' "${SEEDS[*]}")"
  REMOTE_CMD+=" bash -s"
  if ! COMPLETED_RAW="$(
    ssh "${SSH_OPTS[@]}" "$SERVER" "$REMOTE_CMD" <<'REMOTE'
target="$TARGET"
expected="$EXPECTED"
arms="$ARMS"
seeds="$SEEDS"
for arm in $arms; do
  ok=1
  for seed in $seeds; do
    csv="$target/results/runs/sacm_stc/$arm/seed_$seed/eval/tables/all_metrics.csv"
    manifest="$target/results/runs/sacm_stc/$arm/seed_$seed/eval/run_manifest.json"
    if [ ! -f "$csv" ]; then
      echo "  ${arm} seed ${seed}: missing eval/tables/all_metrics.csv" >&2
      ok=0
      continue
    fi
    if [ ! -f "$manifest" ]; then
      echo "  ${arm} seed ${seed}: missing eval/run_manifest.json" >&2
      ok=0
      continue
    fi
    if grep -Eq '"status"[[:space:]]*:[[:space:]]*"(running|failed)"' "$manifest"; then
      echo "  ${arm} seed ${seed}: eval status running/failed" >&2
      ok=0
    fi
    if grep -q '"checkpoint"' "$manifest" \
       && ! grep -q 'last_model.pth' "$manifest"; then
      echo "  ${arm} seed ${seed}: eval checkpoint is not last_model.pth" >&2
      ok=0
    fi
    rows="$(awk 'END { print NR - 1 }' "$csv" 2>/dev/null || echo 0)"
    if ! [[ "$rows" =~ ^[0-9]+$ ]] || [ "$rows" -lt "$expected" ]; then
      echo "  ${arm} seed ${seed}: only ${rows}/${expected} dataset rows" >&2
      ok=0
    fi
  done
  if [ "$ok" -eq 1 ]; then
    printf '%s\n' "$arm"
  fi
done
REMOTE
  )"; then
    echo "ERROR: probe over SSH failed (remote bash/awk error)." >&2
    echo "Inspect: ssh $SERVER 'ls $TARGET/results/runs/sacm_stc'" >&2
    exit 3
  fi
  while IFS= read -r line; do
    [ -n "$line" ] && COMPLETED+=("$line")
  done <<< "$COMPLETED_RAW"
fi

if [ "${#COMPLETED[@]}" -eq 0 ]; then
  echo "ERROR: SSH works, but no arm has a complete ${EXPECTED_DATASETS}-dataset eval." >&2
  echo "The per-seed reasons are printed above.  If you know the completed arms," >&2
  echo "retry with:" >&2
  echo "  $0 --no-probe --arms \"<arm1> <arm2> ...\" --analyze" >&2
  exit 4
fi

echo ""
echo "Completed arms:"
for arm in "${COMPLETED[@]}"; do
  echo "  ${arm}"
done
if [ "$NO_PROBE" -eq 0 ]; then
  for arm in "${ARMS[@]}"; do
    if ! contains_arm "$arm" "${COMPLETED[@]}"; then
      echo "  skip ${arm} (no complete ${EXPECTED_DATASETS}-dataset eval yet)"
    fi
  done
fi

MODEL_SPECS=()
for arm in "${COMPLETED[@]}"; do
  MODEL_SPECS+=("sacm_stc:${arm}")
done
MODEL_LIST="${MODEL_SPECS[*]}"
SEED_LIST="${SEEDS[*]}"

PULL_ARGS=(--models "$MODEL_LIST" --seeds "$SEED_LIST")
if [ "$DRY_RUN" -eq 1 ]; then
  PULL_ARGS+=(--dry-run)
fi

echo ""
echo "=== pulling analysis artifacts ==="
bash "$REPO/scripts/pull_results.sh" "${PULL_ARGS[@]}"

if [ "$ANALYZE" -eq 1 ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "dry-run: would analyze arms: ${COMPLETED[*]}"
  else
    echo ""
    echo "=== local paired analysis ==="
    bash "$REPO/scripts/run_sacm_stc_ablation.sh" --stage analyze \
      --arms "${COMPLETED[@]}" --seeds "${SEEDS[@]}"
    echo ""
    echo "Report: results/ablations/sacm_stc/analysis/ablation_report.md"
  fi
fi

echo ""
echo "Done."
echo "Local status: bash scripts/pull_results.sh --status --models \"$MODEL_LIST\" --seeds \"$SEED_LIST\""
