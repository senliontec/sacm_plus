#!/usr/bin/env bash
# Quick same-protocol test for SACM-X's own model.
#
# 18-image train_root, all 20 frozen-protocol test sets, ONE preset,
# no TTA, W&B on, fast validation, then aggregate the 4 paper metrics.
#
# Usage:
#   GPU=1 PRESET=full EPOCHS=200 bash scripts/run_sacm_x_quick.sh
#   GPU=1 PRESET=sacm EPOCHS=20 bash scripts/run_sacm_x_quick.sh
#   GPU=1 PRESET=full EPOCHS=5 DATASETS="CHASEDB1 WIRE" \
#     bash scripts/run_sacm_x_quick.sh
#
# Environment:
#   PYTHON, GPU, PRESET, EPOCHS, VAL_INTERVAL, SEED, DATASETS, CKPT,
#   TRAIN_ROOT, OUTPUT_ROOT, MONITOR, USE_WANDB, WANDB_MODE,
#   INIT_STAGE2_FROM_STAGE1, SKIP_TEST
#
# MONITOR=official (default): official SACM-style loss/F1/LR terminal lines,
#   no FOV in validation or test, and only loss/F1 during training validation.
# MONITOR=sacm: fast validation but still prints the rich wide table (legacy).
# MONITOR=full: re-enable the full validation metric suite (Dice/IoU/clDice/
#   HD95 + AUC/Betti/Topograph engines).
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
GPU="${GPU:-0}"
PRESET="${PRESET:-full}"
source scripts/wandb_env.sh
EPOCHS="${EPOCHS:-200}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
SEED="${SEED:-42}"
DATASETS="${DATASETS:-}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/sacm_x_quick}"
MONITOR="${MONITOR:-official}"
INIT_STAGE2_FROM_STAGE1="${INIT_STAGE2_FROM_STAGE1:-false}"
SKIP_TEST="${SKIP_TEST:-false}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE
if [ "$USE_WANDB" = "true" ] && [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled (mode=$WANDB_MODE) but no credentials were found." >&2
    echo "Set WANDB_API_KEY or ~/.netrc, or use WANDB_MODE=offline / USE_WANDB=false." >&2
    exit 1
  fi
fi
case "$MONITOR" in
  official) TRAIN_EXTRA=(--use_wandb "$USE_WANDB" --fast_val true
              --val_auc false --val_betti_matching false --val_topograph false
              --log_style official --no-fov)
            EVAL_EXTRA="--use_wandb $USE_WANDB --no-fov" ;;
  sacm) TRAIN_EXTRA=(--use_wandb "$USE_WANDB" --fast_val true
          --val_auc false --val_betti_matching false --val_topograph false)
        EVAL_EXTRA="--use_wandb $USE_WANDB" ;;
  full) TRAIN_EXTRA=(--use_wandb "$USE_WANDB")
        EVAL_EXTRA="--use_wandb $USE_WANDB" ;;
  *) echo "ERROR: MONITOR must be 'official', 'sacm', or 'full', got '$MONITOR'" >&2
     exit 2 ;;
esac

if [ "$INIT_STAGE2_FROM_STAGE1" = "true" ]; then
  TRAIN_EXTRA+=(--init_stage2_from_stage1 true)
fi

if [ ! -f "$CKPT" ]; then
  echo "ERROR: missing SAM ViT-L checkpoint: $CKPT" >&2
  exit 2
fi

SPECS="$("$PYTHON" - "$DATASETS" <<'PY'
import csv
import sys
from pathlib import Path

repo = Path.cwd()
selected = set(sys.argv[1].split()) if sys.argv[1].strip() else None
roots = {}
with open(repo / 'data' / 'sacm_3shot' / 'protocol_manifest.csv') as f:
    for row in csv.DictReader(f):
        if not row['split'].endswith('_test'):
            continue
        name = row['dataset']
        if selected and name not in selected:
            continue
        mask = Path(row['mask_path'])
        if not mask.is_absolute():
            mask = repo / mask
        roots[name] = mask.parent.parent
if not roots:
    raise SystemExit('no test datasets selected')
print(' '.join(f'{name}:{roots[name]}' for name in sorted(roots)))
PY
)"
read -r -a spec_array <<< "$SPECS"

echo "preset=$PRESET epochs=$EPOCHS gpu=$GPU monitor=$MONITOR init_stage2_from_stage1=$INIT_STAGE2_FROM_STAGE1 skip_test=$SKIP_TEST"
echo "datasets=${#spec_array[@]}  output=$OUTPUT_ROOT"
RUN_EXTRA=()
if [ "$SKIP_TEST" = "true" ]; then
  RUN_EXTRA+=(--skip_test)
fi
"$PYTHON" scripts/run_experiments.py \
  --gpus "$GPU" \
  --checkpoint "$CKPT" \
  --train_root "$TRAIN_ROOT" \
  --test_datasets "${spec_array[@]}" \
  --presets "$PRESET" \
  --output_root "$OUTPUT_ROOT" \
  --epochs "$EPOCHS" --val_interval "$VAL_INTERVAL" --seed "$SEED" \
  --extra_train_args "${TRAIN_EXTRA[*]}" \
  --extra_eval_args "$EVAL_EXTRA" \
  ${RUN_EXTRA[@]+"${RUN_EXTRA[@]}"}

if [ "$SKIP_TEST" = "true" ]; then
  echo "== test/eval skipped (SKIP_TEST=true) =="
  echo "checkpoint/log dir: $OUTPUT_ROOT/$PRESET"
  exit 0
fi

echo "== aggregate =="
"$PYTHON" scripts/aggregate_results.py \
  --root "$OUTPUT_ROOT" --out "$OUTPUT_ROOT/tables"

echo "== macro =="
cat "$OUTPUT_ROOT/tables/macro_metrics.csv"
