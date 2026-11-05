#!/usr/bin/env bash
# One-click pull of the completed final + SACM-official result trees.
#
# Run this LOCALLY (not on the server), from the repo root:
#
#   bash scripts/tools/pull_final_official_results.sh
#
# Defaults:
#   REMOTE=zkyd@zkyd
#   REMOTE_ROOT=~/code/sacm-plus/results/runs
#   FINAL_RUN=final_gpu0_4_20261104_153901
#   MODE=metrics   # excludes *.pth, checkpoints/, predictions/, images
#
# Useful overrides:
#   FINAL_RUN=final_gpu0_4_<other> bash scripts/tools/pull_final_official_results.sh
#   MODE=all bash scripts/tools/pull_final_official_results.sh   # include checkpoints
#   DRY_RUN=1 bash scripts/tools/pull_final_official_results.sh
#   REMOTE=user@host REMOTE_ROOT=~/code/sacm-plus/results/runs \
#     bash scripts/tools/pull_final_official_results.sh
#
# Output:
#   results/pulled_analysis_<timestamp>/
#     final/<FINAL_RUN>/...          summary_seeds.csv + component_eval + logs
#     sacm_official/default/seed_*/  eval tables, metadata, train logs
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

REMOTE="${REMOTE:-zkyd@zkyd}"
REMOTE_ROOT="${REMOTE_ROOT:-~/code/sacm-plus/results/runs}"
FINAL_RUN="${FINAL_RUN:-final_gpu0_4_20261104_153901}"
MODE="${MODE:-metrics}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOCAL_ROOT="${LOCAL_ROOT:-results/pulled_analysis_${STAMP}}"
SSH_PORT="${SSH_PORT:-22}"
SSH_CIPHER="${SSH_CIPHER:-aes128-gcm@openssh.com}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
DRY_RUN="${DRY_RUN:-0}"

case "$MODE" in
  metrics)
    FILTERS=(
      --exclude='*.pth'
      --exclude='*.pt'
      --exclude='*.ckpt'
      --exclude='*.safetensors'
      --exclude='checkpoints/'
      --exclude='predictions/'
      --exclude='visualizations/'
      --exclude='*.png'
      --exclude='*.jpg'
      --exclude='*.jpeg'
      --exclude='*.bmp'
      --exclude='*.tif'
      --exclude='*.tiff'
      --exclude='*.npy'
      --exclude='*.npz'
    )
    ;;
  all)
    FILTERS=()
    ;;
  *)
    echo "ERROR: MODE must be metrics | all (got '$MODE')" >&2
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

mkdir -p "$LOCAL_ROOT/final" "$LOCAL_ROOT/sacm_official"

echo "============================================================"
echo " Pull final + SACM official results"
echo " remote: $REMOTE:$REMOTE_ROOT"
echo " final:  $FINAL_RUN"
echo " mode:   $MODE"
echo " local:  $LOCAL_ROOT"
echo " dry:    $DRY_RUN"
echo "============================================================"

# Final component-repair run: summary_seeds.csv, per-seed summaries,
# classifier/per-dataset CSVs, training logs, run metadata.  Checkpoints and
# prediction images are excluded in metrics mode.
"$RSYNC_BIN" "${RSYNC_OPTS[@]}" "${FILTERS[@]}" \
  -e "$SSH_CMD" \
  "$REMOTE:$REMOTE_ROOT/$FINAL_RUN/" \
  "$LOCAL_ROOT/final/$FINAL_RUN/"

# SACM official: macro/all metric tables plus per-dataset eval artifacts and
# run metadata for seeds 0-2.  Predictions and .pth files are excluded in
# metrics mode.
"$RSYNC_BIN" "${RSYNC_OPTS[@]}" "${FILTERS[@]}" \
  -e "$SSH_CMD" \
  "$REMOTE:$REMOTE_ROOT/sacm_official/default/" \
  "$LOCAL_ROOT/sacm_official/default/"

cat > "$LOCAL_ROOT/README.txt" <<EOF
Pulled from: $REMOTE:$REMOTE_ROOT
Final run:   $FINAL_RUN
Mode:        $MODE
Time:        $STAMP

Key files:
  final/$FINAL_RUN/summary_seeds.csv
  final/$FINAL_RUN/component_eval/seed_*/summary.json
  final/$FINAL_RUN/component_eval/seed_*/per_dataset.csv
  sacm_official/default/seed_*/eval/tables/macro_metrics.csv
  sacm_official/default/seed_*/eval/tables/all_metrics.csv
  sacm_official/default/seed_*/train.log
EOF

echo
echo "============================================================"
echo " Done"
echo " local size:  $(du -sh "$LOCAL_ROOT" | awk '{print $1}')"
echo " local files: $(find "$LOCAL_ROOT" -type f | wc -l)"
echo " final summary: $LOCAL_ROOT/final/$FINAL_RUN/summary_seeds.csv"
echo " official macro: $LOCAL_ROOT/sacm_official/default/seed_0/eval/tables/macro_metrics.csv"
echo " local root:  $LOCAL_ROOT"
echo "============================================================"
