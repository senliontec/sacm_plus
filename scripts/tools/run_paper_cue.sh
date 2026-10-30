#!/usr/bin/env bash
# One-click cue-ml pipeline: train queue -> eval queue -> report/figures.
#
# Usage:
#   bash scripts/tools/run_paper_cue.sh --seeds 0
#   bash scripts/tools/run_paper_cue.sh --models unet,dscnet --seeds 0 --dry-run
#   bash scripts/tools/run_paper_cue.sh --models samed,dscnet --seeds 0 --no-resume
#
# cue schedules one job per GPU and assigns each job via
# CUDA_VISIBLE_DEVICES.  It does not detect already-running processes, so the
# script refuses to start while any GPU has compute processes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODELS="unet,cs2net,bcunet,maskvsc,nnwnet,jtfn,dscnet,segdino,samed,topolora,sam_adapter,sam_octa,cwsam,sacm_official,persam,knowsam,sam2_unet,sam_med2d,ucs,matcher"
SEEDS="0"
PROTOCOL="seeded5"
DRY_RUN=0
ALLOW_BUSY=0
NO_RESUME=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --models) MODELS="${2:?missing value for --models}"; shift 2 ;;
    --seeds) SEEDS="${2:?missing value for --seeds}"; shift 2 ;;
    --protocol) PROTOCOL="${2:?missing value for --protocol}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-busy) ALLOW_BUSY=1; shift ;;
    --no-resume) NO_RESUME=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
CUE="${CUE:-$(command -v cue || true)}"
if [ -z "$CUE" ] && [ -x "$(dirname "$PYTHON")/cue" ]; then
  CUE="$(dirname "$PYTHON")/cue"
fi
if [ -z "$CUE" ]; then
  echo "ERROR: cue is not installed; pip install cue-ml" >&2
  exit 1
fi
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: Python is not executable: $PYTHON" >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -f "$ROOT/scripts/tools/wandb_env.sh" ]; then
  # Reuse the project's W&B host/entity/key; non-fatal if unavailable.
  source "$ROOT/scripts/tools/wandb_env.sh" || true
fi

if [ "$DRY_RUN" -eq 0 ] && [ "$ALLOW_BUSY" -eq 0 ]; then
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
          | sed '/^[[:space:]]*$/d' || true)"
  if [ -n "$busy" ]; then
    echo "ERROR: GPU compute processes are still running:" >&2
    echo "$busy" >&2
    echo "Stop them or pass --allow-busy (unsafe: cue does not check VRAM)." >&2
    exit 1
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
CUE_ROOT="$ROOT/results/cue"
TRAIN_YAML="$CUE_ROOT/train_${PROTOCOL}_${STAMP}.yaml"
EVAL_YAML="$CUE_ROOT/eval_${PROTOCOL}_${STAMP}.yaml"
RESUME_ARGS=()
if [ "$NO_RESUME" -eq 1 ]; then
  RESUME_ARGS+=(--no-resume)
fi

"$PYTHON" "$ROOT/scripts/tools/make_cue_config.py" \
  --stage train --models "$MODELS" --seeds $SEEDS \
  --protocol "$PROTOCOL" --out "$TRAIN_YAML" \
  --python-cmd "$PYTHON" --log-dir "$CUE_ROOT/logs" \
  "${RESUME_ARGS[@]}"
"$PYTHON" "$ROOT/scripts/tools/make_cue_config.py" \
  --stage eval --models "$MODELS" --seeds $SEEDS \
  --protocol "$PROTOCOL" --out "$EVAL_YAML" \
  --python-cmd "$PYTHON" --log-dir "$CUE_ROOT/logs" \
  "${RESUME_ARGS[@]}"

run_cue() {
  local yaml="$1"
  if [ "$DRY_RUN" -eq 1 ]; then
    "$CUE" -p "$yaml" --dry-run
  else
    # cue waits for ENTER for its confirmation prompt.
    printf '\n' | "$CUE" -p "$yaml" --fail-fast
  fi
}

run_cue "$TRAIN_YAML"
run_cue "$EVAL_YAML"

if [ "$DRY_RUN" -eq 0 ]; then
  bash "$ROOT/scripts/run_report.sh" \
    --model "$MODELS" --seeds $SEEDS --protocol "$PROTOCOL"
  if [ -f "$ROOT/results/paper/tables/all_metrics.csv" ]; then
    bash "$ROOT/scripts/run_figures.sh"
  fi
fi
