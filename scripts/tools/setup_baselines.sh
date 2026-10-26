#!/usr/bin/env bash
# One-click dependencies for the self-contained baseline zoo.
#
# Torch/torchvision are assumed to be installed by the user (CUDA build).
# This installs the project in editable mode plus every baseline extra.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi

"$PYTHON" -m pip install -e \
  '.[maskvsc,bcunet,cwsam,sam_med2d,sam_octa,knowsam,nnwnet,segdino,matcher]'

echo "Installed. Next:"
echo "  export PYTHONPATH=$PWD/src"
echo "  $PYTHON -m baseline_zoo doctor"
echo "  $PYTHON -m baseline_zoo plan --model all --seeds all --gpus 0,1"
