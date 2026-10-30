#!/usr/bin/env bash
# One-click paper pipeline: train -> evaluate -> tables -> figures.
#
# Usage:
#   GPU=1 bash scripts/run_paper.sh --model unet --seeds 0
#   bash scripts/run_paper.sh --model all --seeds all --gpus 0,1,2,3 --resume
#   bash scripts/run_paper.sh --model unet --seeds 0 --gpus 2
#   bash scripts/run_paper.sh --model all --seeds all --gpus 0 --no-kill
#   bash scripts/run_paper.sh --model all --seeds all --tables-only
#   bash scripts/run_paper.sh --model all --seeds all --dry-run
#
# GPU selection:
#   --gpus is authoritative.  GPU=<indices> is also accepted when --gpus is
#   omitted; setting both to different values is an error.  This prevents a
#   stale GPU=... environment variable from silently selecting another card.
#
# GPU cleanup:
#   The selected GPU(s) are ALWAYS cleared before training, so a stale run
#   can never share the card.  Only the GPUs passed via --gpus/GPU are
#   touched.  Pass --no-kill to skip cleanup (e.g. when sharing a card);
#   --kill-gpu is retained as a no-op for backward compatibility.
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
# Force-clear the selected GPU(s) before every train/eval run.  --no-kill
# is the explicit opt-out; --kill-gpu remains accepted for compatibility.
KILL_GPU=1
GPU_ENV="${GPU:-}"
GPU_FLAG=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --tables-only|--report-only) TABLES_ONLY=1 ;;
    --kill-gpu) KILL_GPU=1 ;;
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
      if [ $((idx + 1)) -ge "${#ARGS[@]}" ] \
          || [ -z "${ARGS[$((idx + 1))]:-}" ]; then
        echo "[run-paper] ERROR: --gpus requires a value (e.g. 0 or 0,1)" >&2
        exit 2
      fi
      GPU_SPEC="${ARGS[$((idx + 1))]}"
      GPU_FLAG=1
      idx=$((idx + 2))
      ;;
    --gpus=*)
      GPU_SPEC="${arg#--gpus=}"
      if [ -z "$GPU_SPEC" ]; then
        echo "[run-paper] ERROR: --gpus= requires a value (e.g. 0 or 0,1)" >&2
        exit 2
      fi
      GPU_FLAG=1
      idx=$((idx + 1))
      ;;
    *)
      idx=$((idx + 1))
      ;;
  esac
done

GPU_SPEC="${GPU_SPEC// /}"
if ! [[ "$GPU_SPEC" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "[run-paper] ERROR: invalid GPU spec: $GPU_SPEC" >&2
  exit 2
fi

if [ -n "$GPU_ENV" ]; then
  GPU_ENV="${GPU_ENV// /}"
  if ! [[ "$GPU_ENV" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[run-paper] ERROR: invalid GPU environment value: GPU=$GPU_ENV" >&2
    exit 2
  fi
  if [ "$GPU_FLAG" -eq 1 ]; then
    if [ "$GPU_ENV" != "$GPU_SPEC" ]; then
      echo "[run-paper] ERROR: GPU=$GPU_ENV conflicts with --gpus $GPU_SPEC." >&2
      echo "[run-paper] Use either 'GPU=$GPU_ENV bash ...' or '--gpus $GPU_ENV', not both with different values." >&2
      exit 2
    fi
  else
    GPU_SPEC="$GPU_ENV"
    if [ "$TABLES_ONLY" -eq 0 ]; then
      ARGS+=(--gpus "$GPU_SPEC")
    fi
    echo "[run-paper] using GPU=$GPU_SPEC from environment"
  fi
fi

if [ "$TABLES_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  if [ "$KILL_GPU" -eq 1 ]; then
    bash scripts/tools/clear_gpus.sh --gpus "$GPU_SPEC"
  else
    echo "[run-paper] GPU cleanup skipped by --no-kill; leaving GPU(s) $GPU_SPEC untouched"
  fi
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
