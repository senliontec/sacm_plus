#!/usr/bin/env bash
# One-click U-Net training on the frozen 3-shot split.
#
# Usage:
#   bash scripts/run_unet_train.sh                    # background by default
#   BACKGROUND=false bash scripts/run_unet_train.sh   # foreground
#   EPOCHS=50 bash scripts/run_unet_train.sh          # shorter run
#   USE_WANDB=false bash scripts/run_unet_train.sh    # no W&B
#   WANDB_MODE=offline bash scripts/run_unet_train.sh # local W&B logs
#
# Environment:
#   PYTHON, EPOCHS, BATCH_SIZE, LR, INPUT_SIZE, SEED, DEVICE,
#   WORKDIR, LOG_DIR, USE_WANDB, WANDB_MODE, WANDB_PROJECT,
#   WANDB_ENTITY, WANDB_GROUP, WANDB_NAME, WANDB_TAGS,
#   WANDB_BASE_URL, WANDB_API_KEY
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
source scripts/wandb_env.sh

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"
INPUT_SIZE="${INPUT_SIZE:-512}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
WORKDIR="${WORKDIR:-workdirs/baselines/unet}"
LOG_DIR="${LOG_DIR:-logs}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"
WANDB_GROUP="${WANDB_GROUP:-baselines}"

mkdir -p "$LOG_DIR" "$WORKDIR"

# Fail loudly instead of training silently without the requested W&B run.
if [ "$USE_WANDB" = "true" ] && [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled (mode=$WANDB_MODE) but no credentials were found." >&2
    echo "Expected ~/.netrc credentials for ${WANDB_BASE_URL}, or WANDB_API_KEY." >&2
    echo "Use WANDB_MODE=offline for a local run." >&2
    exit 1
  fi
fi

LOG="${_UNET_LOG:-$LOG_DIR/unet_train_$(date +%Y%m%d_%H%M%S).log}"

if [ "${BACKGROUND:-true}" = "true" ] && [ -z "${_UNET_BG_REENTRY:-}" ]; then
  nohup env _UNET_BG_REENTRY=1 _UNET_LOG="$LOG" BACKGROUND=false \
    bash "$SELF" "$@" >"$LOG" 2>&1 &
  echo "U-Net training started in background (pid $!)"
  echo "log: $LOG"
  echo "checkpoint: $WORKDIR/best_model.pth"
  echo "tail: tail -f $LOG"
  exit 0
fi

CMD=("$PYTHON" baselines/adapters/unet.py --stage train
     --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR"
     --input-size "$INPUT_SIZE" --seed "$SEED" --device "$DEVICE"
     --workdir "$WORKDIR")
if [ "$USE_WANDB" = "true" ]; then
  CMD+=(--use_wandb --wandb_project "$WANDB_PROJECT"
        --wandb_mode "$WANDB_MODE" --wandb_group "$WANDB_GROUP")
  [ -n "$WANDB_ENTITY" ] && CMD+=(--wandb_entity "$WANDB_ENTITY")
  [ -n "${WANDB_NAME:-}" ] && CMD+=(--wandb_name "$WANDB_NAME")
  [ -n "${WANDB_TAGS:-}" ] && CMD+=(--wandb_tags "$WANDB_TAGS")
fi

echo "command: ${CMD[*]}"
echo "log: $LOG"
if [ -n "${_UNET_LOG:-}" ]; then
  # Background parent already redirects stdout/stderr to this log.
  "${CMD[@]}"
else
  "${CMD[@]}" 2>&1 | tee "$LOG"
fi
