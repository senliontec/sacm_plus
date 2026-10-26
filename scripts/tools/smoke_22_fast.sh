#!/usr/bin/env bash
# Fast environment smoke test for all 22 baselines.
#
# Each model is forced to one epoch/iteration and evaluated on one DRIVE
# image only.  This does not modify configs/models/*.yaml and does not run
# the full 20-dataset paper metrics.  Use it to verify that build, train,
# checkpoint save, model load, and prediction all work.
#
# Usage:
#   bash scripts/tools/smoke_22_fast.sh
#   GPU=7 bash scripts/tools/smoke_22_fast.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY=${PYTHON:-}
if [ -z "$PY" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PY=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PY=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PY=python
  fi
fi
if [ ! -x "$PY" ]; then
  echo "ERROR: Python not executable: $PY" >&2
  exit 1
fi
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

GPU=${GPU:-7}
CLEAR_GPU=${CLEAR_GPU:-1}
SMOKE_ROOT=$(mktemp -d /tmp/sacm_smoke_fast.XXXXXX)
mkdir -p "$SMOKE_ROOT/data/DRIVE/images" "$SMOKE_ROOT/data/DRIVE/masks"

# Make a one-image DRIVE test root from the seed-0 protocol.
"$PY" - "$REPO_ROOT/data/sacm_3shot_seed0/protocol_manifest.csv" \
  "$SMOKE_ROOT/data/DRIVE" <<'PY'
import csv
import os
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out = Path(sys.argv[2])
row = next(r for r in csv.DictReader(manifest.open())
           if r['dataset'] == 'DRIVE' and r['split'] == 'seen_test')
image = Path(row['image_path'])
mask = Path(row['mask_path'])
os.symlink(image.resolve(), out / 'images' / image.name)
os.symlink(mask.resolve(), out / 'masks' / mask.name)
print(f'smoke query: {image} + {mask}')
PY
if [ "$?" -ne 0 ]; then
  echo "ERROR: could not create one-image smoke dataset" >&2
  exit 1
fi

MODELS=(
  unet cs2net bcunet maskvsc nnwnet jtfn dscnet segdino samed topolora
  sam_adapter sam_octa cwsam sacm_official fob_sam persam knowsam
  sam2_unet sam_med2d ucs matcher sam_vit_l_unet
)

echo "============================================================"
echo "FAST SMOKE: 22 models, 1 epoch/iteration, 1 test image"
echo "python: $PY"
echo "gpu: $GPU (clear=$CLEAR_GPU)"
echo "smoke root: $SMOKE_ROOT"
echo "============================================================"

"$PY" -u -m baseline_zoo doctor
if [ "$?" -ne 0 ]; then
  echo "ERROR: doctor failed; fix environment before smoke" >&2
  exit 1
fi

if [ "$CLEAR_GPU" = "1" ]; then
  echo "clearing compute processes on GPU $GPU"
  bash scripts/tools/clear_gpus.sh --gpus "$GPU" || exit 1
else
  echo "CLEAR_GPU=0: not clearing GPU $GPU"
fi

failed=()
for m in "${MODELS[@]}"; do
  echo ""
  echo "------------------------------------------------------------"
  echo "SMOKE: $m"
  echo "------------------------------------------------------------"

  # One epoch by default.  KnowSAM uses an iteration budget; FoB-SAM
  # otherwise runs 3000 iterations per "epoch"; training-free models use 0.
  extra=(--epochs 1)
  case "$m" in
    knowsam)
      extra=(--epochs 1 --max-iterations 1 --unlabeled-limit 8)
      ;;
    fob_sam)
      extra=(--epochs 1 --fob-iters-per-epoch 1)
      ;;
    matcher|persam)
      extra=(--epochs 0)
      ;;
  esac

  log="$SMOKE_ROOT/$m.log"
  if "$PY" -m baseline_zoo.run_one \
      --model "$m" \
      --stage all \
      --data-root data/sacm_3shot_seed0 \
      --workdir "$SMOKE_ROOT/runs/$m" \
      --pred-root "$SMOKE_ROOT/preds/$m" \
      --limit-train 8 \
      --limit-val 1 \
      --test-datasets "DRIVE:$SMOKE_ROOT/data/DRIVE" \
      --seed 0 \
      --device "cuda:$GPU" \
      --use_wandb false \
      "${extra[@]}" >"$log" 2>&1; then
    echo "PASS: $m"
  else
    echo "FAIL: $m (log: $log)"
    tail -40 "$log"
    failed+=("$m")
  fi
done

echo ""
echo "============================================================"
echo "FAST SMOKE RESULT"
echo "smoke root: $SMOKE_ROOT"
if [ "${#failed[@]}" -eq 0 ]; then
  echo "ALL 22 PASSED"
else
  printf 'FAILED: %s\n' "${failed[*]}"
  exit 1
fi
