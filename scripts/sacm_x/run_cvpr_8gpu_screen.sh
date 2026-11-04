#!/usr/bin/env bash
# CVPR-time 8-GPU screening run: 5-seed baseline + three method candidates.
#
#   GPU 0: full_fixed2_aux        seed 0   (5-seed baseline)
#   GPU 1: full_fixed2_aux        seed 1
#   GPU 2: full_fixed2_aux        seed 2
#   GPU 3: full_fixed2_aux        seed 3
#   GPU 4: full_fixed2_aux        seed 4
#   GPU 5: full_fixed2_scnp       seed 0   (training-only topology loss)
#   GPU 6: full_fixed2_resmoe     seed 0   (gate + residual experts)
#   GPU 7: full_fixed2_lora_evp   seed 0   (ConvLoRA + EVP hybrid)
#
# Default is training + validation only (300 epochs), no 20-test eval.  The
# second wave (seed 1-4 of the winner, plus EVP-only/ConvLoRA-only controls)
# is launched only after this screen.  After the screen, evaluate only the
# winner with eval_sacm_x_20.sh.
#
# Usage:
#   bash scripts/sacm_x/run_cvpr_8gpu_screen.sh
#   EPOCHS=800 bash scripts/sacm_x/run_cvpr_8gpu_screen.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PY=/home/zkyd/anaconda3/envs/tsai/bin/python
  elif [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PY=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PY=python
  fi
fi

EPOCHS="${EPOCHS:-300}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
SCREEN_ROOT="${SCREEN_ROOT:-results/runs/cvpr8gpu_screen}"
LOG_DIR="${LOG_DIR:-results/ablation_logs}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Never inherit a diagnostic/audit output root.
unset OUTPUT_ROOT
if printf '%s\n' "$SCREEN_ROOT" | grep -qiE 'audit|/eval(/|$)'; then
  echo "ERROR: SCREEN_ROOT must not be an audit/eval directory: $SCREEN_ROOT" >&2
  exit 2
fi

# Required code/files: fail loudly instead of silently running the old model.
missing=()
for f in \
  src/models/sam_network/convlora.py \
  src/models/sam_network/evp.py \
  src/models/sam_network/residual_moe.py \
  src/models/sacm_x/configs.py \
  src/core/trainer.py \
  src/train/trainer.py \
  scripts/sacm_x/run_full_fixed2_aux.sh \
  scripts/sacm_x/run_full_fixed2_scnp.sh \
  scripts/sacm_x/run_full_fixed2_resmoe.sh \
  scripts/sacm_x/run_full_fixed2_lora_evp.sh; do
  [ -f "$f" ] || missing+=("$f")
done
for p in full_fixed2_aux full_fixed2_scnp full_fixed2_resmoe \
         full_fixed2_lora_evp; do
  grep -q "\"$p\"" src/models/sacm_x/configs.py || missing+=("preset:$p")
done
grep -q "moe_loss_weight" src/core/trainer.py || missing+=("trainer MoE loss")
grep -q "loss_gate" src/core/trainer.py || missing+=("trainer ResMoE loss")
grep -q "use_convlora" src/train/trainer.py || missing+=("trainer ConvLoRA args")
grep -q "use_res_moe" src/train/trainer.py || missing+=("trainer ResMoE args")
grep -q "residual_moe" src/models/sam_network/mask_decoder.py || missing+=("decoder ResMoE")
grep -q "evp_zero_init" src/models/sam_network/build_sam.py || missing+=("builder EVP args")
grep -q "use_res_moe" src/models/sam_network/build_sam.py || missing+=("builder ResMoE args")
grep -q "dilation=dilation" src/models/sam_network/convlora.py || missing+=("convlora dilation fix")
[ -f "$CKPT" ] || missing+=("$CKPT")
for s in 0 1 2 3 4; do
  [ -f "data/sacm_3shot_seed${s}/protocol_manifest.csv" ] \
    || missing+=("seed${s} data")
done
if [ "${#missing[@]}" -gt 0 ]; then
  printf 'ERROR: missing prerequisites:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "Sync the current working tree and data before running this screen." >&2
  exit 2
fi

# All eight cards must be free enough for a training job.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found" >&2
  exit 2
fi
if [ "$(nvidia-smi -L | wc -l)" -lt 8 ]; then
  echo "ERROR: fewer than 8 GPUs visible" >&2
  exit 2
fi
for gpu in 0 1 2 3 4 5 6 7; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  if [[ "${used:-0}" =~ ^[0-9]+$ ]] && [ "$used" -gt 2000 ]; then
    echo "ERROR: GPU $gpu already uses ${used} MiB; free it first." >&2
    exit 2
  fi
done

echo "============================================================"
echo " CVPR 8-GPU baseline + SCNP/ResMoE/hybrid screen"
echo " root:    $SCREEN_ROOT"
echo " epochs:  $EPOCHS (train+val only; no 20-test eval)"
echo " wandb:   $USE_WANDB ($WANDB_MODE)"
echo " ckpt:    $CKPT"
echo " python:  $PY"
echo " git:     $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "============================================================"

PIDS=()
LABELS=()
LOGS=()

run_one() {
  local gpu="$1" preset="$2" seed="$3"
  local log="$LOG_DIR/cvpr8gpu_${preset}_seed${seed}_${STAMP}.log"
  echo "GPU $gpu  ->  $preset / seed $seed  ($log)"
  PYTHON="$PY" GPU="$gpu" SEED="$seed" EPOCHS="$EPOCHS" \
  VAL_INTERVAL="$VAL_INTERVAL" CKPT="$CKPT" \
  TRAIN_ROOT="data/sacm_3shot_seed${seed}" \
  CANONICAL_ROOT="$SCREEN_ROOT" OUTPUT_ROOT= SCNP=false \
  USE_WANDB="$USE_WANDB" WANDB_MODE="$WANDB_MODE" \
  NUM_WORKERS="$NUM_WORKERS" \
  WANDB_NAME="sacm_x_cvpr8_${preset}_seed${seed}" \
    bash "scripts/sacm_x/run_${preset}.sh" train \
    > "$log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("${preset}/seed${seed}/gpu${gpu}")
  LOGS+=("$log")
}

run_one 0 full_fixed2_aux          0
run_one 1 full_fixed2_aux          1
run_one 2 full_fixed2_aux          2
run_one 3 full_fixed2_aux          3
run_one 4 full_fixed2_aux          4
run_one 5 full_fixed2_scnp         0
run_one 6 full_fixed2_resmoe       0
run_one 7 full_fixed2_lora_evp     0

RC=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "DONE  ${LABELS[$i]}"
  else
    echo "FAIL  ${LABELS[$i]}  (see ${LOGS[$i]})"
    tail -n 20 "${LOGS[$i]}" || true
    RC=1
  fi
done

echo
echo "============================================================"
echo " Screening finished (rc=$RC)"
echo " Best val checkpoints / metrics are under:"
echo "   $SCREEN_ROOT/sacm_x/<preset>/seed_<seed>/"
echo " Logs:"
echo "   $LOG_DIR/cvpr8gpu_*_${STAMP}.log"
if [ "$RC" -eq 0 ]; then
  echo
  echo "Next: inspect metrics_history.json for each run, then evaluate only"
  echo "the winner on 20 test sets, e.g.:"
  echo "  GPU=<free> SEED=0 PRESET=<winning_preset> \\"
  echo "  TRAIN_OUT=$SCREEN_ROOT/sacm_x/<winning_preset>/seed_0 \\"
  echo "  bash scripts/sacm_x/eval_sacm_x_20.sh"
fi
echo "============================================================"
exit "$RC"
