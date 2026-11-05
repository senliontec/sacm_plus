#!/usr/bin/env bash
# Pull only the five final SACM-X checkpoints from the server.
#
# Run LOCALLY from the repo root:
#   bash scripts/tools/pull_final_checkpoints.sh
#
# Defaults:
#   REMOTE=zkyd@zkyd
#   REMOTE_ROOT=~/code/sacm-plus/results/runs
#   FINAL_RUN=final_gpu0_4_20261104_153901
#   PRESET=full_fixed2_gate
#   SEEDS="0 1 2 3 4"
#   LOCAL_ROOT=results/pulled_checkpoints/<FINAL_RUN>
#
# Local layout mirrors the server:
#   <LOCAL_ROOT>/train/sacm_x/<preset>/seed_<s>/best_model.pth
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

REMOTE="${REMOTE:-zkyd@zkyd}"
REMOTE_ROOT="${REMOTE_ROOT:-~/code/sacm-plus/results/runs}"
FINAL_RUN="${FINAL_RUN:-final_gpu0_4_20261104_153901}"
PRESET="${PRESET:-full_fixed2_gate}"
SEEDS="${SEEDS:-0 1 2 3 4}"
LOCAL_ROOT="${LOCAL_ROOT:-results/pulled_checkpoints/${FINAL_RUN}}"
SSH_PORT="${SSH_PORT:-22}"
SSH_CIPHER="${SSH_CIPHER:-aes128-gcm@openssh.com}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
DRY_RUN="${DRY_RUN:-0}"

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

echo "============================================================"
echo " Pull final SACM-X checkpoints"
echo " remote: $REMOTE:$REMOTE_ROOT/$FINAL_RUN"
echo " preset: $PRESET"
echo " seeds:  $SEEDS"
echo " local:  $LOCAL_ROOT"
echo " dry:    $DRY_RUN"
echo "============================================================"

for seed in $SEEDS; do
  remote_path="$REMOTE_ROOT/$FINAL_RUN/train/sacm_x/$PRESET/seed_${seed}/best_model.pth"
  local_path="$LOCAL_ROOT/train/sacm_x/$PRESET/seed_${seed}/best_model.pth"
  mkdir -p "$(dirname "$local_path")"
  echo "[seed $seed] $REMOTE:$remote_path"
  "$RSYNC_BIN" "${RSYNC_OPTS[@]}" \
    -e "$SSH_CMD" \
    "$REMOTE:$remote_path" "$local_path"
done

echo
echo "============================================================"
echo " Done"
echo " local size:  $(du -sh "$LOCAL_ROOT" | awk '{print $1}')"
echo " checkpoints:"
find "$LOCAL_ROOT" -name 'best_model.pth' -printf '  %p  %s bytes\n' | sort
echo "============================================================"
