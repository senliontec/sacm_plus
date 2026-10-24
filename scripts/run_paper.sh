#!/usr/bin/env bash
# One-click paper pipeline: train -> evaluate -> tables -> figures.
#
# Usage:
#   bash scripts/run_paper.sh --model all --seeds all --gpus 0,1,2,3 --resume
#   bash scripts/run_paper.sh --model unet --seeds 0 --gpus 2
#   bash scripts/run_paper.sh --model all --seeds all --tables-only
#   bash scripts/run_paper.sh --model all --seeds all --dry-run
#
# --tables-only skips training/evaluation and only regenerates the paper
# tables and figures from existing metrics.
#
# Stage wrappers: run_train.sh, run_eval.sh, run_report.sh, run_figures.sh.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

TABLES_ONLY=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --tables-only|--report-only) TABLES_ONLY=1 ;;
    *) ARGS+=("$arg") ;;
  esac
done

DRY_RUN=0
for arg in "${ARGS[@]}"; do
  [ "$arg" = "--dry-run" ] && DRY_RUN=1
done

if [ "$TABLES_ONLY" -eq 0 ]; then
  bash scripts/run_train.sh ${ARGS[@]+"${ARGS[@]}"}
  bash scripts/run_eval.sh ${ARGS[@]+"${ARGS[@]}"}
else
  echo "tables-only: skipping train/eval"
fi

bash scripts/run_report.sh ${ARGS[@]+"${ARGS[@]}"}
if [ "$DRY_RUN" -eq 0 ] && [ -f results/paper/tables/all_metrics.csv ]; then
  bash scripts/run_figures.sh
fi
