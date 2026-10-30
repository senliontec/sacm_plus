#!/usr/bin/env bash
# One-click cue-ml pipeline: train queue -> eval queue -> report/figures.
#
# Usage:
#   bash scripts/tools/run_paper_cue.sh --seeds 0
#   bash scripts/tools/run_paper_cue.sh --models unet,dscnet --seeds 0 --dry-run
#   bash scripts/tools/run_paper_cue.sh --models samed,dscnet --seeds 0 --no-resume
#   bash scripts/tools/run_paper_cue.sh --seeds "0 1 2 3 4" --order seed
#
# Queue order defaults to a strict seed barrier: for seed 0, train all models
# to completion, then evaluate all models; only then start seed 1.  Pass
# --order model for the older all-train-then-all-eval, longest-first order.
#
# cue schedules one job per GPU and assigns each job via
# CUDA_VISIBLE_DEVICES.  It does not detect already-running processes, so the
# script refuses to start while any GPU has compute processes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODELS="unet,cs2net,bcunet,maskvsc,nnwnet,jtfn,dscnet,segdino,samed,topolora,sam_adapter,sam_octa,cwsam,sacm_official,persam,knowsam,sam2_unet,sam_med2d,ucs,matcher,sam_vit_l_unet"
SEEDS="0"
PROTOCOL="seeded5"
DRY_RUN=0
ALLOW_BUSY=0
NO_RESUME=0
ORDER="seed"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --models) MODELS="${2:?missing value for --models}"; shift 2 ;;
    --seeds) SEEDS="${2:?missing value for --seeds}"; shift 2 ;;
    --protocol) PROTOCOL="${2:?missing value for --protocol}"; shift 2 ;;
    --order) ORDER="${2:?missing value for --order}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-busy) ALLOW_BUSY=1; shift ;;
    --no-resume) NO_RESUME=1; shift ;;
    -h|--help)
      sed -n '2,16p' "$0"
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
if [ -f "$ROOT/scripts/sacm_x/wandb_env.sh" ]; then
  # Reuse the project's W&B host/entity/key; non-fatal if unavailable.
  source "$ROOT/scripts/sacm_x/wandb_env.sh" || true
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
RESUME_ARGS=()
if [ "$NO_RESUME" -eq 1 ]; then
  RESUME_ARGS+=(--no-resume)
fi
case "$ORDER" in
  model|seed) ;;
  *) echo "ERROR: --order must be 'model' or 'seed', got '$ORDER'" >&2
     exit 2 ;;
esac

make_config() {
  local stage="$1" seeds="$2" out="$3"
  "$PYTHON" "$ROOT/scripts/tools/make_cue_config.py" \
    --stage "$stage" --models "$MODELS" --seeds $seeds \
    --protocol "$PROTOCOL" --order "$ORDER" --out "$out" \
    --python-cmd "$PYTHON" --log-dir "$CUE_ROOT/logs" \
    "${RESUME_ARGS[@]}"
}

run_cue() {
  local yaml="$1"
  if [ "$DRY_RUN" -eq 1 ]; then
    "$CUE" -p "$yaml" --dry-run
  else
    # cue waits for ENTER for its confirmation prompt.
    printf '\n' | "$CUE" -p "$yaml" --fail-fast
  fi
}

if [ "$ORDER" = "seed" ]; then
  # Strict barrier: finish train+eval for one seed before starting the next.
  if [ "$SEEDS" = "all" ]; then
    SEEDS="$("$PYTHON" - "$PROTOCOL" <<'PY'
import sys
from baseline_zoo.protocols import load_protocol
print(*load_protocol(sys.argv[1]).seeds)
PY
)"
  fi
  for seed in $SEEDS; do
    TRAIN_YAML="$CUE_ROOT/train_${PROTOCOL}_seed${seed}_${STAMP}.yaml"
    EVAL_YAML="$CUE_ROOT/eval_${PROTOCOL}_seed${seed}_${STAMP}.yaml"
    make_config train "$seed" "$TRAIN_YAML"
    run_cue "$TRAIN_YAML"
    make_config eval "$seed" "$EVAL_YAML"
    run_cue "$EVAL_YAML"
  done
else
  TRAIN_YAML="$CUE_ROOT/train_${PROTOCOL}_${STAMP}.yaml"
  EVAL_YAML="$CUE_ROOT/eval_${PROTOCOL}_${STAMP}.yaml"
  make_config train "$SEEDS" "$TRAIN_YAML"
  make_config eval "$SEEDS" "$EVAL_YAML"
  run_cue "$TRAIN_YAML"
  run_cue "$EVAL_YAML"
fi

if [ "$DRY_RUN" -eq 0 ]; then
  bash "$ROOT/scripts/run_report.sh" \
    --model "$MODELS" --seeds $SEEDS --protocol "$PROTOCOL"
  if [ -f "$ROOT/results/paper/tables/all_metrics.csv" ]; then
    bash "$ROOT/scripts/run_figures.sh"
  fi
fi
