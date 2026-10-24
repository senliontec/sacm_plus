#!/usr/bin/env bash
# One-click paper pipeline: train -> evaluate -> aggregate tables.
#
# Usage:
#   bash scripts/run_paper.sh --model all --seeds all --gpus 0,1,2,3 --resume
#   bash scripts/run_paper.sh --model unet --seeds 0 --gpus 2
#   bash scripts/run_paper.sh --model all --seeds all --dry-run
#
# Report-only regeneration is available via scripts/run_report.sh.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

bash scripts/run_train.sh "$@"
bash scripts/run_eval.sh "$@"
bash scripts/run_report.sh "$@"
if [ -f results/paper/tables/all_metrics.csv ]; then
  bash scripts/run_figures.sh
fi
