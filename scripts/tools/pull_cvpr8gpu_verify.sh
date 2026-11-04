#!/usr/bin/env bash
# One-click pull of the new cvpr8gpu_verify screen from the server.
#
# Run LOCALLY from the repo root:
#   bash scripts/tools/pull_cvpr8gpu_verify.sh
#
# Default MODE=metadata: pulls metrics, configs, provenance, screen index and
# logs, but NOT .pth checkpoints and NOT PNGs.
#
#   MODE=metadata  run metadata + screen files + logs
#   MODE=all       everything, including best/last/checkpoints
#   DRY_RUN=1      show what would be transferred
#
# Defaults:
#   REMOTE=zkyd@172.16.1.7
#   REMOTE_ROOT=~/code/sacm-plus/results/runs/cvpr8gpu_verify
#   LOCAL_ROOT=results/runs/cvpr8gpu_verify
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

REMOTE="${REMOTE:-zkyd@172.16.1.7}"
REMOTE_ROOT="${REMOTE_ROOT:-~/code/sacm-plus/results/runs/cvpr8gpu_verify}"
LOCAL_ROOT="${LOCAL_ROOT:-results/runs/cvpr8gpu_verify}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-~/code/sacm-plus/results/ablation_logs}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-results/ablation_logs}"
MODE="${MODE:-metadata}"
DRY_RUN="${DRY_RUN:-0}"
SSH_PORT="${SSH_PORT:-22}"
SSH_CIPHER="${SSH_CIPHER:-aes128-gcm@openssh.com}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"

case "$MODE" in
  metadata|all) ;;
  *)
    echo "ERROR: MODE must be metadata | all (got '$MODE')" >&2
    exit 2
    ;;
esac

if ! command -v "$RSYNC_BIN" >/dev/null 2>&1; then
  echo "ERROR: rsync not found" >&2
  exit 2
fi

SSH_CMD="ssh -T -p ${SSH_PORT} -o Compression=no \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
if [ -n "$SSH_CIPHER" ]; then
  SSH_CMD+=" -c ${SSH_CIPHER}"
fi

RSYNC_OPTS=(
  -a
  --no-owner
  --no-group
  --partial
  --info=progress2
  --human-readable
  --timeout=120
  --no-motd
)
if [ "$MODE" = "all" ]; then
  # Resume large, immutable files only in full mode.  Never use this while
  # training is rewriting last_model.pth.
  RSYNC_OPTS+=(--append-verify)
fi
if [ "$DRY_RUN" = "1" ]; then
  RSYNC_OPTS+=(--dry-run)
fi

IMAGE_EXCLUDES=(
  --exclude='*.png'
  --exclude='*.jpg'
  --exclude='*.jpeg'
  --exclude='*.bmp'
  --exclude='*.tif'
  --exclude='*.tiff'
  --exclude='*.npy'
  --exclude='*.npz'
)

if [ "$MODE" = "metadata" ]; then
  # Whitelist metadata instead of blacklisting weights.  This is critical:
  # during training, last_model.pth is rewritten every epoch, and pulling it
  # with --append-verify can grow forever / show >100% progress.
  RUN_FILTERS=(
    --exclude='checkpoints/'
    --exclude='eval/'
    --include='*/'
    --include='metrics_history.json'
    --include='train_config.json'
    --include='train_config.resume_*.json'
    --include='run_info.json'
    --include='train.log'
    --include='wandb_run.json'
    --include='checkpoint.json'
    --include='provenance/***'
    "${IMAGE_EXCLUDES[@]}"
    --exclude='*'
  )
else
  RUN_FILTERS=()
fi

mkdir -p "$LOCAL_ROOT" "$LOCAL_LOG_DIR"

echo "============================================================"
echo " Pull cvpr8gpu_verify screen"
echo " remote: $REMOTE:$REMOTE_ROOT"
echo " local:  $LOCAL_ROOT"
echo " mode:   $MODE"
echo " dry:    $DRY_RUN"
echo "============================================================"

# 1) Run directories (metadata or all).  This is the required payload: a
# connection/auth failure must not be reported as a successful pull.
if ! "$RSYNC_BIN" "${RSYNC_OPTS[@]}" "${RUN_FILTERS[@]}" \
    -e "$SSH_CMD" \
    "$REMOTE:$REMOTE_ROOT/sacm_x/" "$LOCAL_ROOT/sacm_x/"; then
  echo "ERROR: could not pull run directories from $REMOTE:$REMOTE_ROOT/sacm_x" >&2
  echo "Check SSH access, then rerun this script." >&2
  exit 1
fi

# 2) Screen-level files: jobs.csv, code_version.txt, code.patch,
#    untracked_files.txt.  Do not recurse into sacm_x again.
if ! "$RSYNC_BIN" "${RSYNC_OPTS[@]}" \
    --exclude='sacm_x/' \
    -e "$SSH_CMD" \
    "$REMOTE:$REMOTE_ROOT/" "$LOCAL_ROOT/"; then
  echo "ERROR: could not pull screen-level files from $REMOTE:$REMOTE_ROOT" >&2
  exit 1
fi

# 3) Launcher/training logs.  Missing/partial logs are non-fatal because the
# screen may still be running.
"$RSYNC_BIN" "${RSYNC_OPTS[@]}" \
  --include='cvpr8gpu_verify_*' --exclude='*' \
  -e "$SSH_CMD" \
  "$REMOTE:$REMOTE_LOG_DIR/" "$LOCAL_LOG_DIR/" || true

if [ "$(find "$LOCAL_ROOT" -type f 2>/dev/null | wc -l)" -eq 0 ]; then
  echo "ERROR: no files were pulled." >&2
  exit 1
fi

echo
echo "============================================================"
echo " Pulled"
echo " runs:  $LOCAL_ROOT/sacm_x"
echo " index: $LOCAL_ROOT/jobs.csv"
echo " code:  $LOCAL_ROOT/code_version.txt"
echo " logs:  $LOCAL_LOG_DIR/cvpr8gpu_verify_*"
echo " files: $(find "$LOCAL_ROOT" -type f 2>/dev/null | wc -l)"
echo " size:  $(du -sh "$LOCAL_ROOT" 2>/dev/null | awk '{print $1}')"
echo "============================================================"
