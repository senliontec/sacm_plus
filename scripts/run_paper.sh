#!/usr/bin/env bash
# One-click paper pipeline: train -> evaluate -> tables -> figures.
#
# Usage:
#   bash scripts/run_paper.sh --model all --seeds all --gpus 0,1,2,3 --resume
#   bash scripts/run_paper.sh --model unet --seeds 0 --gpus 2
#   bash scripts/run_paper.sh --model all --seeds all --tables-only
#   bash scripts/run_paper.sh --model all --seeds all --dry-run
#   bash scripts/run_paper.sh --model all --seeds all --no-kill
#
# Before training, compute processes on the GPUs selected by --gpus are
# cleared.  --no-kill skips that cleanup; other GPUs are never touched.
#
# --tables-only skips training/evaluation and only regenerates the paper
# tables and figures from existing metrics.
#
# Stage wrappers: run_train.sh, run_eval.sh, run_report.sh, run_figures.sh.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

# The paper server is offline.  Fail fast instead of waiting on HuggingFace
# timeouts; all pretrained assets must be present under checkpoints/.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

TABLES_ONLY=0
KILL_GPU=1
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --tables-only|--report-only) TABLES_ONLY=1 ;;
    --no-kill) KILL_GPU=0 ;;
    *) ARGS+=("$arg") ;;
  esac
done

DRY_RUN=0
for arg in "${ARGS[@]}"; do
  [ "$arg" = "--dry-run" ] && DRY_RUN=1
done

# Match the planner's default GPU set.  Accept --gpus 0 and --gpus=0.
GPU_SPEC="0"
idx=0
while [ "$idx" -lt "${#ARGS[@]}" ]; do
  arg="${ARGS[$idx]}"
  case "$arg" in
    --gpus)
      GPU_SPEC="${ARGS[$((idx + 1))]:-0}"
      idx=$((idx + 2))
      ;;
    --gpus=*)
      GPU_SPEC="${arg#--gpus=}"
      idx=$((idx + 1))
      ;;
    *)
      idx=$((idx + 1))
      ;;
  esac
done

if [ "$TABLES_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ] && [ "$KILL_GPU" -eq 1 ]; then
  bash scripts/tools/clear_gpus.sh --gpus "$GPU_SPEC"
fi

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
