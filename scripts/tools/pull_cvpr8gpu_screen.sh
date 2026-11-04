#!/usr/bin/env bash
# Fast one-click pull of the old 8-GPU screen from the server.
#
# Run this LOCALLY (not on the server), from the repo root:
#
#   bash scripts/tools/pull_cvpr8gpu_screen.sh
#
# Modes:
#   MODE=all         every file, including best/last/checkpoints and images
#   MODE=metadata    no *.pth, no checkpoints/, no images (tables/provenance/logs)
#   MODE=eval        only eval/ trees, no images (fastest for table analysis)
#   MODE=eval-images only eval/ trees, including result PNGs
#
# Examples:
#   MODE=eval bash scripts/tools/pull_cvpr8gpu_screen.sh
#   MODE=eval-images bash scripts/tools/pull_cvpr8gpu_screen.sh
#   MODE=metadata bash scripts/tools/pull_cvpr8gpu_screen.sh
#   MODE=all DRY_RUN=1 bash scripts/tools/pull_cvpr8gpu_screen.sh
#
# Defaults match the server prompt:
#   REMOTE=zkyd@zkyd
#   REMOTE_ROOT=~/code/sacm-plus/results/runs/cvpr8gpu_screen/sacm_x
#   LOCAL_ROOT=results/runs/cvpr8gpu_screen/sacm_x
#
# rsync is used with whole-file mode, partial resume and no SSH compression:
# the tree is dominated by .pth files, which do not compress.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

REMOTE="${REMOTE:-zkyd@zkyd}"
REMOTE_ROOT="${REMOTE_ROOT:-~/code/sacm-plus/results/runs/cvpr8gpu_screen/sacm_x}"
LOCAL_ROOT="${LOCAL_ROOT:-results/runs/cvpr8gpu_screen/sacm_x}"
MODE="${MODE:-all}"
DRY_RUN="${DRY_RUN:-0}"
SSH_PORT="${SSH_PORT:-22}"
SSH_CIPHER="${SSH_CIPHER:-aes128-gcm@openssh.com}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"

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

case "$MODE" in
  all)
    FILTERS=()
    ;;
  metadata)
    FILTERS=(
      --exclude='*.pth'
      --exclude='checkpoints/'
      "${IMAGE_EXCLUDES[@]}"
    )
    ;;
  eval)
    # First match wins: exclude images before the eval include rule, then
    # include eval files, then drop everything outside eval/.  The image
    # files are per-image result PNGs, not needed for metric analysis.
    FILTERS=(
      --include='*/'
      "${IMAGE_EXCLUDES[@]}"
      --include='*/eval/***'
      --exclude='*'
    )
    ;;
  eval-images)
    FILTERS=(--include='*/' --include='*/eval/***' --exclude='*')
    ;;
  *)
    echo "ERROR: MODE must be all | metadata | eval | eval-images (got '$MODE')" >&2
    exit 2
    ;;
esac

if ! command -v "$RSYNC_BIN" >/dev/null 2>&1; then
  echo "ERROR: rsync not found; install rsync or set RSYNC_BIN." >&2
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
  --whole-file
  --partial
  --append-verify
  --info=progress2
  --human-readable
  --timeout=120
  --no-motd
)
if [ "$DRY_RUN" = "1" ]; then
  RSYNC_OPTS+=(--dry-run)
fi

mkdir -p "$LOCAL_ROOT"

echo "============================================================"
echo " Pull old 8-GPU SACM-X screen"
echo " remote: $REMOTE:$REMOTE_ROOT"
echo " local:  $LOCAL_ROOT"
echo " mode:   $MODE"
echo " cipher: ${SSH_CIPHER:-ssh default}"
echo " dry:    $DRY_RUN"
echo "============================================================"

"$RSYNC_BIN" "${RSYNC_OPTS[@]}" "${FILTERS[@]}" \
  -e "$SSH_CMD" \
  "$REMOTE:$REMOTE_ROOT/" "$LOCAL_ROOT/"

echo
echo "============================================================"
echo " Done"
echo " local size: $(du -sh "$LOCAL_ROOT" | awk '{print $1}')"
echo " local files: $(find "$LOCAL_ROOT" -type f | wc -l)"
echo " local root:  $LOCAL_ROOT"
echo "============================================================"
