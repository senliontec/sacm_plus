#!/usr/bin/env bash
# One-command overnight launcher for the SACM-STC scope/structure ablation.
#
# It self-daemonizes, then:
#   1. audits the existing 800-epoch all_dense checkpoints (no training);
#   2. runs the 6-arm x 4-seed x 300-epoch matrix on the requested GPUs;
#   3. each 300-epoch run is evaluated on its best_model.pth over the
#      20 test datasets, then candidate diagnostics run;
#   4. the generic runner writes the paired analysis report at the end.
#
# Usage (server, repo root):
#   bash scripts/run_sacm_stc_ablation_overnight.sh
#   bash scripts/run_sacm_stc_ablation_overnight.sh --gpus 4,5,6,7
#   bash scripts/run_sacm_stc_ablation_overnight.sh --skip-audit
#
# Defaults:
#   --gpus 4,5,6,7
#   seeds 0,1,2,3 (all six arms)
#   config configs/ablations/sacm_stc_scope_structure.yaml
#
# Result separation:
#   training/eval: results/runs/sacm_stc/<arm>/seed_<s>/
#   diagnostics:   results/ablations/sacm_stc/candidate_diagnostics/<arm>/seed_<s>/
#   audit:         results/ablations/sacm_stc/candidate_audit/seed_<s>/
#   analysis:      results/ablations/sacm_stc/analysis/
#   driver log:    results/logs/sacm_stc_ablation/overnight_<stamp>.log
# The existing results/runs/sacm_stc/all_dense tree is read by the audit only
# and is never written by this launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
cd "$REPO"

CONFIG="configs/ablations/sacm_stc_scope_structure.yaml"
# Server defaults for the ablation queue: only GPUs 4-7 are reserved for
# this matrix; GPUs 0-3 may be busy with other jobs.  --gpus still overrides.
GPU_LIST="${GPU_LIST:-4,5,6,7}"
SEEDS=(0 1 2 3)
SKIP_AUDIT=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus)
      GPU_LIST="${2:?--gpus needs a comma-separated list}"
      shift 2
      ;;
    --gpus=*)
      GPU_LIST="${1#--gpus=}"
      shift
      ;;
    --config)
      CONFIG="${2:?--config needs a path}"
      shift 2
      ;;
    --config=*)
      CONFIG="${1#--config=}"
      shift
      ;;
    --skip-audit)
      SKIP_AUDIT=1
      shift
      ;;
    -h|--help)
      sed -n '1,35p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: missing config: $CONFIG" >&2
  exit 2
fi

if ! grep -q -- '--stage2-loss-scope' \
    "$REPO/src/baseline_zoo/adapters/sacm_stc.py"; then
  echo "ERROR: server code is stale: sacm_stc.py has no --stage2-loss-scope." >&2
  echo "Sync the current repository before launching the ablation." >&2
  exit 2
fi
if [ ! -f "$REPO/checkpoints/sam_vit_l_0b3195.pth" ]; then
  echo "ERROR: missing SAM ViT-L checkpoint: checkpoints/sam_vit_l_0b3195.pth" >&2
  exit 2
fi

# Fail fast if a requested GPU is absent or already occupied.  The training
# runner itself has no wait-for-GPU logic; discovering this at 2 a.m. would
# waste the night.
if command -v nvidia-smi >/dev/null 2>&1 \
   && [ "${ALLOW_BUSY:-0}" != "1" ]; then
  MAX_USED_MIB="${MAX_USED_MIB:-2000}"
  IFS=',' read -r -a _ablation_gpus <<< "$GPU_LIST"
  for _gpu in "${_ablation_gpus[@]}"; do
    _used="$(nvidia-smi -i "$_gpu" \
      --query-gpu=memory.used --format=csv,noheader,nounits \
      2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
    if ! [[ "$_used" =~ ^[0-9]+$ ]]; then
      echo "ERROR: GPU $_gpu is not visible to nvidia-smi." >&2
      exit 2
    fi
    if [ "$_used" -gt "$MAX_USED_MIB" ]; then
      echo "ERROR: GPU $_gpu already uses ${_used} MiB;" \
           "threshold MAX_USED_MIB=${MAX_USED_MIB}." >&2
      echo "Free the GPUs or set ALLOW_BUSY=1 to override." >&2
      exit 2
    fi
  done
  unset _ablation_gpus _gpu _used
fi

LOG_DIR="$REPO/results/logs/sacm_stc_ablation"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/overnight_${STAMP}.log"
PID_FILE="$LOG_DIR/overnight_${STAMP}.pid"

# Refuse a second simultaneous overnight matrix: two drivers on the same
# four GPUs would OOM or silently overwrite each other's logs.  The lock FD
# is inherited by the daemon child and released when that child exits.
LOCK_FILE="$LOG_DIR/overnight.lock"
LOCKED=0
if command -v flock >/dev/null 2>&1; then
  if [ "${SACM_ABLATION_LOCKED:-0}" = "1" ]; then
    # The daemon child inherits FD 9 and the parent's flock; reopening the
    # lock file here would race the parent and could fail before it exits.
    LOCKED=1
  else
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
      echo "ERROR: another overnight ablation is already running." >&2
      echo "       lock: $LOCK_FILE" >&2
      exit 2
    fi
    printf '%s\n' "$$" >&9
    LOCKED=1
    export SACM_ABLATION_LOCKED=1
  fi
fi

CONFIG_EPOCHS="$(sed -n \
  's/^epochs:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$CONFIG" | head -n 1)"
if [ "${CONFIG_EPOCHS:-}" != "300" ]; then
  echo "ERROR: config epochs=${CONFIG_EPOCHS:-unknown}; this launcher" \
       "requires exactly 300." >&2
  exit 2
fi

# Use the same interpreter discovery as the one-click runners before sourcing
# W&B, so wandb_auth_ok() checks the environment that will actually train.
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

# Fast W&B preflight before spending time on the audit.  The runner wrapper
# checks again; this only prevents a silent late failure after training starts.
source "$REPO/scripts/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
if [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled but no credentials were found." >&2
    echo "Set WANDB_API_KEY or run 'wandb login', or use WANDB_MODE=offline/disabled." >&2
    exit 1
  fi
fi

# Self-daemonize: the user gets a prompt back immediately; the child keeps
# running after logout and writes everything to the timestamped log.
if [ "${SACM_ABLATION_DAEMON:-0}" != "1" ]; then
  SACM_ABLATION_DAEMON=1 nohup bash "$SCRIPT_PATH" \
    --config "$CONFIG" --gpus "$GPU_LIST" \
    $([ "$SKIP_AUDIT" -eq 1 ] && echo --skip-audit || true) \
    >"$LOG" 2>&1 &
  PID=$!
  echo "$PID" >"$PID_FILE"
  echo "Ablation overnight job launched."
  echo "  pid:  $PID"
  echo "  log:  $LOG"
  echo "  pid file: $PID_FILE"
  echo "  training/eval: results/runs/sacm_stc/<arm>/seed_<s>/"
  echo "  diagnostics:   results/ablations/sacm_stc/candidate_diagnostics/"
  echo "  analysis:      results/ablations/sacm_stc/analysis/"
  echo "The existing all_dense main runs are not modified."
  exit 0
fi

cd "$REPO"
echo "============================================================"
echo " SACM-STC overnight ablation"
echo " start: $(date -Is)"
echo " host:  $(hostname)"
echo " gpus:  $GPU_LIST"
echo " seeds: ${SEEDS[*]}"
echo " config: $CONFIG"
echo " audit: $([ "$SKIP_AUDIT" -eq 1 ] && echo skipped || echo enabled)"
echo " log:   $LOG"
echo "============================================================"

# Save the exact matrix used for this overnight run alongside the analysis
# artifacts.  Training checkpoints also carry train_config.json.
mkdir -p "$REPO/results/ablations/sacm_stc"
cp "$CONFIG" \
  "$REPO/results/ablations/sacm_stc/config_snapshot_${STAMP}.yaml"

if [ "$SKIP_AUDIT" -eq 0 ]; then
  echo "[1/2] candidate audit of existing all_dense checkpoints"
  # The audit is diagnostic.  A missing/old audit checkpoint must not block
  # the actual training matrix; record the failure and continue.
  if ! bash scripts/run_sacm_stc_ablation.sh \
      --config "$CONFIG" --stage audit \
      --seeds "${SEEDS[@]}" --gpus "$GPU_LIST"; then
    echo "[warn] candidate audit failed; continuing with the full matrix"
    echo "[warn] inspect results/ablations/sacm_stc/candidate_audit/"
  fi
else
  echo "[1/2] candidate audit skipped"
fi

echo "[2/2] training + 20-dataset eval + candidate diagnostics"
if bash scripts/run_sacm_stc_ablation.sh \
    --config "$CONFIG" --stage all \
    --seeds "${SEEDS[@]}" --gpus "$GPU_LIST"; then
  MATRIX_STATUS=0
else
  MATRIX_STATUS=$?
  echo "[warn] matrix exited with status ${MATRIX_STATUS}; completed pairs" \
       "were still analyzed. Inspect per-job logs below." >&2
fi

echo "============================================================"
echo " SACM-STC overnight ablation finished"
echo " end: $(date -Is)"
echo " status: ${MATRIX_STATUS} (0 = all jobs complete)"
echo " logs: results/logs/sacm_stc_ablation/"
echo " report: results/ablations/sacm_stc/analysis/ablation_report.md"
echo "============================================================"
exit "$MATRIX_STATUS"
