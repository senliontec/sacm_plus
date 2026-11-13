#!/usr/bin/env bash
# One-click pull of the completed SACM-GCD ablation arms.
#
# It probes the server first and only pulls arms whose seeds each have a
# complete 20-dataset eval/tables/all_metrics.csv and a completed training
# run_info.json.  This avoids treating a still-training arm as a result.
#
# The actual transfer is delegated to scripts/pull_results.sh, so it uses
# the same SSH/rsync retry and include/exclude rules (no predictions or
# model weights by default).
#
# Usage (local repo root):
#   bash scripts/pull_sacm_gcd_ablation.sh
#   bash scripts/pull_sacm_gcd_ablation.sh --analyze
#   bash scripts/pull_sacm_gcd_ablation.sh --dry-run
#   bash scripts/pull_sacm_gcd_ablation.sh --seeds "0 1 2 3"
#   bash scripts/pull_sacm_gcd_ablation.sh --arms "b0_selected_base ss_selected_dense_selected_structure"
#   bash scripts/pull_sacm_gcd_ablation.sh \
#     --config configs/ablations/sacm_gcd_scope_structure.yaml \
#     --arms "b0_selected_base b3_all_dense_all_structure" --analyze
#
# Options:
#   --config PATH          experiment YAML (default: scope_structure)
#   --checkpoint-name NAME override config eval_checkpoint for the probe
#   --no-probe             pull requested arms without server checks
#
# If --arms is omitted, the arm list is read from the selected config.
#
# Env overrides:
#   SERVER=zkyd@172.16.1.7
#   TARGET=/home/zkyd/code/sacm-plus
#   EXPECTED_DATASETS=20
#   EXPECTED_EPOCHS=300
#
# After the pull, --analyze runs the selected config:
#   bash scripts/run_sacm_gcd_ablation.sh --stage analyze \
#     --config <config> --arms <completed arms> --seeds <seeds>
# The checkpoint name, epoch budget and analysis root default to the config's
# ``eval_checkpoint``, ``epochs`` and ``analysis.output_root``.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SERVER="${SERVER:-zkyd@172.16.1.7}"
TARGET="${TARGET:-/home/zkyd/code/sacm-plus}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"
CONFIG="${CONFIG:-configs/ablations/sacm_gcd_scope_structure.yaml}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"
EXPECTED_EPOCHS="${EXPECTED_EPOCHS:-}"
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

ARMS=()
ARMS_GIVEN=0
SEEDS=(0 1 2 3)
ANALYZE=0
DRY_RUN=0
NO_PROBE=0

usage() {
  sed -n '2,39p' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --arms)
      [ "$#" -ge 2 ] || { echo "missing value for --arms" >&2; exit 2; }
      shift
      IFS=', ' read -r -a ARMS <<< "$1"
      ARMS_GIVEN=1
      ;;
    --seeds)
      [ "$#" -ge 2 ] || { echo "missing value for --seeds" >&2; exit 2; }
      shift
      IFS=', ' read -r -a SEEDS <<< "$1"
      ;;
    --analyze) ANALYZE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --config)
      [ "$#" -ge 2 ] || { echo "missing value for --config" >&2; exit 2; }
      shift
      CONFIG="$1"
      ;;
    --checkpoint-name)
      [ "$#" -ge 2 ] || {
        echo "missing value for --checkpoint-name" >&2; exit 2; }
      shift
      CHECKPOINT_NAME="$1"
      ;;
    --no-probe) NO_PROBE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "ERROR: --seeds is empty" >&2
  exit 2
fi
if ! [[ "$EXPECTED_DATASETS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EXPECTED_DATASETS must be an integer" >&2
  exit 2
fi
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi
if [ -z "$CHECKPOINT_NAME" ]; then
  CHECKPOINT_NAME="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
config = yaml.safe_load(open(sys.argv[1])) or {}
print(config.get('eval_checkpoint', 'best_model.pth') or '')
PY
)" || { echo "ERROR: cannot read eval_checkpoint from $CONFIG" >&2; exit 2; }
fi
if [ -z "$EXPECTED_EPOCHS" ]; then
  EXPECTED_EPOCHS="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
config = yaml.safe_load(open(sys.argv[1])) or {}
print(config.get('epochs', '') or '')
PY
)" || { echo "ERROR: cannot read epochs from $CONFIG" >&2; exit 2; }
fi
ANALYSIS_ROOT="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
config = yaml.safe_load(open(sys.argv[1])) or {}
print((config.get('analysis') or {}).get(
    'output_root', 'results/ablations/sacm_gcd/analysis') or '')
PY
)" || { echo "ERROR: cannot read analysis.output_root" >&2; exit 2; }
if [ -z "$CHECKPOINT_NAME" ]; then
  echo "ERROR: config has no eval_checkpoint and --checkpoint-name is empty" >&2
  exit 2
fi
if [ -n "$EXPECTED_EPOCHS" ] && ! [[ "$EXPECTED_EPOCHS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EXPECTED_EPOCHS must be an integer" >&2
  exit 2
fi
if [ "${#ARMS[@]}" -eq 0 ] && [ "$ARMS_GIVEN" -eq 0 ]; then
  ARMS_TEXT="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
config = yaml.safe_load(open(sys.argv[1])) or {}
for arm in (config.get('arms') or {}):
    print(arm)
PY
)" || { echo "ERROR: cannot read arms from $CONFIG" >&2; exit 2; }
  ARMS=()
  while IFS= read -r arm; do
    [ -n "$arm" ] && ARMS+=("$arm")
  done <<< "$ARMS_TEXT"
fi
if [ "${#ARMS[@]}" -eq 0 ]; then
  echo "ERROR: --arms is empty and config has no arms" >&2
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
  REMOTE_CMD+=" CHECKPOINT=$(printf '%q' "$CHECKPOINT_NAME")"
  REMOTE_CMD+=" EPOCHS=$(printf '%q' "$EXPECTED_EPOCHS")"
  REMOTE_CMD+=" ARMS=$(printf '%q' "${ARMS[*]}")"
  REMOTE_CMD+=" SEEDS=$(printf '%q' "${SEEDS[*]}")"
  REMOTE_CMD+=" bash -s"
  if ! COMPLETED_RAW="$(
    ssh "${SSH_OPTS[@]}" "$SERVER" "$REMOTE_CMD" <<'REMOTE'
target="$TARGET"
expected="$EXPECTED"
checkpoint_name="$CHECKPOINT"
expected_epochs="$EPOCHS"
arms="$ARMS"
seeds="$SEEDS"
for arm in $arms; do
  ok=1
  for seed in $seeds; do
    csv="$target/results/runs/sacm_gcd/$arm/seed_$seed/eval/tables/all_metrics.csv"
    manifest="$target/results/runs/sacm_gcd/$arm/seed_$seed/eval/run_manifest.json"
    run_info="$target/results/runs/sacm_gcd/$arm/seed_$seed/run_info.json"
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
    if [ ! -f "$run_info" ]; then
      echo "  ${arm} seed ${seed}: missing run_info.json" >&2
      ok=0
      continue
    fi
    if ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"done"' "$run_info"; then
      echo "  ${arm} seed ${seed}: training status is not done" >&2
      ok=0
    fi
    if [ -n "$expected_epochs" ] \
       && ! grep -Eq '"epochs"[[:space:]]*:[[:space:]]*'"$expected_epochs"'([^0-9]|$)' \
            "$run_info"; then
      echo "  ${arm} seed ${seed}: training epochs != ${expected_epochs}" >&2
      ok=0
    fi
    if grep -Eq '"status"[[:space:]]*:[[:space:]]*"(running|failed)"' "$manifest"; then
      echo "  ${arm} seed ${seed}: eval status running/failed" >&2
      ok=0
    fi
    if [ -n "$checkpoint_name" ] && grep -q '"checkpoint"' "$manifest" \
       && ! grep -Fq "$checkpoint_name" "$manifest"; then
      echo "  ${arm} seed ${seed}: eval checkpoint is not ${checkpoint_name}" >&2
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
    echo "Inspect: ssh $SERVER 'ls $TARGET/results/runs/sacm_gcd'" >&2
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
  MODEL_SPECS+=("sacm_gcd:${arm}")
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
    bash "$REPO/scripts/run_sacm_gcd_ablation.sh" --stage analyze \
      --config "$CONFIG" \
      --arms "${COMPLETED[@]}" --seeds "${SEEDS[@]}"
    echo ""
    echo "Report: ${ANALYSIS_ROOT}/ablation_report.md"
  fi
fi

echo ""
echo "Done."
echo "Local status: bash scripts/pull_results.sh --status --models \"$MODEL_LIST\" --seeds \"$SEED_LIST\""
