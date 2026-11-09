#!/usr/bin/env bash
# Archive the legacy SACM-STC main result trees before a clean rerun.
#
# SACM-STC used to have two sibling trees:
#   results/runs/sacm_stc/all_dense/   original training artifacts
#   results/runs/sacm_stc/default/     historical eval alias/mirror
# This script moves BOTH to a timestamped archive so a clean rerun has one
# canonical main tree: results/runs/sacm_stc/all_dense/.
#
# Ablation arm directories such as b0_selected_base or b2_* are deliberately
# left untouched.
#
# Usage (server or local, repo root):
#   bash scripts/archive_sacm_stc_main.sh
#   DRY_RUN=1 bash scripts/archive_sacm_stc_main.sh
#   STAMP=pre_rerun bash scripts/archive_sacm_stc_main.sh
#
# Archives:
#   results/archive/sacm_stc/<stamp>/all_dense/
#   results/archive/sacm_stc/<stamp>/default/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

SRC="$REPO/results/runs/sacm_stc"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DEST_ROOT="${DEST_ROOT:-$REPO/results/archive/sacm_stc}"
DEST="$DEST_ROOT/$STAMP"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -d "$SRC" ]; then
  echo "ERROR: missing source tree: $SRC" >&2
  exit 2
fi

found=0
for name in all_dense default; do
  path="$SRC/$name"
  if [ ! -d "$path" ]; then
    echo "skip: $path does not exist"
    continue
  fi
  found=1
  if [ "$DRY_RUN" = "1" ]; then
    echo "dry-run: mv $path -> $DEST/$name"
    continue
  fi
  mkdir -p "$DEST"
  if [ -e "$DEST/$name" ]; then
    echo "ERROR: archive target already exists: $DEST/$name" >&2
    exit 2
  fi
  mv "$path" "$DEST/$name"
  echo "archived: $path -> $DEST/$name"
done

if [ "$found" -eq 0 ]; then
  echo "nothing to archive under $SRC"
  exit 0
fi
if [ "$DRY_RUN" = "1" ]; then
  echo "dry-run complete; no files moved"
else
  echo "archive complete: $DEST"
  echo "canonical rerun target: $SRC/all_dense"
  echo "SACM-STC no longer has a default/ alias under results/runs."
fi
