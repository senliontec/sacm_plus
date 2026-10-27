#!/usr/bin/env bash
# One-click: run all 21 paper baselines for seed 0 on GPU 0, sequentially.
#
# Usage:
#   bash scripts/tools/run_21_baselines_seed0_gpu0.sh
#
# The list is explicit so --model all (which also includes sacm_x) is not
# used.  run_paper.sh performs train, eval, report, and figures in order.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Prefer the project's tsai environment, but allow an explicit override:
#   PYTHON=/path/to/conda/env/bin/python bash scripts/tools/...
if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: Python is not executable: $PYTHON" >&2
  echo "Set PYTHON=/path/to/conda/env/bin/python and rerun." >&2
  exit 1
fi
export PYTHON
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# 21 baselines, in execution order.  sacm_x is deliberately excluded.
MODELS=(
  unet
  cs2net
  bcunet
  maskvsc
  nnwnet
  jtfn
  dscnet
  segdino
  samed
  topolora
  sam_adapter
  sam_octa
  cwsam
  sacm_official
  persam
  knowsam
  sam2_unet
  sam_med2d
  ucs
  matcher
  sam_vit_l_unet
)
MODELS_CSV=$(IFS=,; echo "${MODELS[*]}")

GPUS=${GPUS:-0}
echo "============================================================"
echo "SACM baseline run: seed 0, GPU(s) ${GPUS}, sequential per GPU"
echo "python: $PYTHON"
echo "models (${#MODELS[@]}): ${MODELS[*]}"
echo "============================================================"

# Fail early if a required checkpoint or Python dependency is missing.
# Run doctor as a stream, not in command substitution: a failing doctor under
# `set -e` would otherwise exit silently and hide its diagnostics.
mkdir -p results
doctor_log="$REPO_ROOT/results/run_21_seed0_gpu0.doctor.log"
set +e
"$PYTHON" -u -m baseline_zoo doctor 2>&1 | tee "$doctor_log"
doctor_status=${PIPESTATUS[0]}
set -e
if [ "$doctor_status" -ne 0 ]; then
  echo "ERROR: preflight doctor failed (exit $doctor_status); see $doctor_log" >&2
  exit 1
fi

echo "Starting train -> eval -> report -> figures..."
exec bash scripts/run_paper.sh \
  --model "$MODELS_CSV" \
  --seeds 0 \
  --gpus "$GPUS" \
  --resume
