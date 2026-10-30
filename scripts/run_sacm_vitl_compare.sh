#!/usr/bin/env bash
# One-click ViT-L comparison on a multi-GPU server:
#   GPU 0: official SACM baseline (sacm_official)
#   GPU 1..N: one project SACM-X preset per GPU
#
# Default: official SACM on GPU0 and SACM-X `full` on GPU1.
# Topology arm: official SACM + SACM-X full + SACM-X full_srec, e.g.
#   SACM_X_PRESETS="full full_srec" bash scripts/run_sacm_vitl_compare.sh
# All presets share:
#   - SAM ViT-L checkpoint
#   - the same seed-specific 3-shot split
#   - the same epoch budget
#   - W&B online
#   - the full 20-dataset test protocol
#
# Usage:
#   bash scripts/run_sacm_vitl_compare.sh
#   SEED=0 EPOCHS=800 bash scripts/run_sacm_vitl_compare.sh
#   SACM_X_PRESETS="full full_srec" bash scripts/run_sacm_vitl_compare.sh
#   SACM_X_PRESETS="full full_srec_low full_srec_high" \
#     SACM_X_GPUS="1 2 3" bash scripts/run_sacm_vitl_compare.sh
#
# Optional environment:
#   SEED=0                  protocol seed (default 0)
#   EPOCHS=200              same budget for every arm (default 200)
#   PRESET=full             single SACM-X preset (legacy single-arm knob)
#   SACM_X_PRESETS=full     space-separated SACM-X presets (default: $PRESET)
#   SACM_X_GPUS="1 2"       optional explicit GPUs for the presets
#   VAL_INTERVAL=1          SACM-X validation interval (default 1)
#   ROOT=results/sacm_vitl_compare/seed0
#   GROUP=sacm_vitl_compare_seed0
#   USE_WANDB=true
#   WANDB_MODE=online
#   ALLOW_BUSY=0            set 1 to bypass the GPU memory guard
#
# Outputs:
#   <ROOT>/sacm_official/{best_model.pth,predictions,paper}
#   <ROOT>/sacm_x/<PRESET>/best_model.pth + predictions/tables
#   <ROOT>/sacm_x/tables/ (aggregated across presets)
#   <ROOT>/logs/{sacm_official_train,sacm_official_eval,sacm_x_<PRESET>}.log
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-200}"
PRESET="${PRESET:-full}"
SACM_X_PRESETS="${SACM_X_PRESETS:-$PRESET}"
SACM_X_GPUS="${SACM_X_GPUS:-}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
ROOT="${ROOT:-results/sacm_vitl_compare/seed${SEED}}"
GROUP="${GROUP:-sacm_vitl_compare_seed${SEED}}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
ALLOW_BUSY="${ALLOW_BUSY:-0}"

DATA_ROOT="${DATA_ROOT:-data/sacm_3shot_seed${SEED}}"
PROTOCOL="${PROTOCOL:-$DATA_ROOT/protocol_manifest.csv}"
SAM_CKPT="${SAM_CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
OFFICIAL_DIR="$ROOT/sacm_official"
SACM_X_DIR="$ROOT/sacm_x"
LOG_DIR="$ROOT/logs"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_RUN_GROUP="$GROUP"
export WANDB_MODE

if [ -f scripts/tools/wandb_env.sh ]; then
  # shellcheck disable=SC1091
  source scripts/tools/wandb_env.sh
elif [ -f scripts/sacm_x/wandb_env.sh ]; then
  # shellcheck disable=SC1091
  source scripts/sacm_x/wandb_env.sh
fi

if [ ! -f "$SAM_CKPT" ]; then
  echo "ERROR: missing SAM ViT-L checkpoint: $SAM_CKPT" >&2
  exit 2
fi
if [ ! -f "$PROTOCOL" ]; then
  echo "ERROR: missing protocol: $PROTOCOL" >&2
  echo "Build the seed-specific 3-shot split first." >&2
  exit 2
fi
if [ ! -f scripts/sacm_x/run_sacm_x_quick.sh ]; then
  echo "ERROR: missing scripts/sacm_x/run_sacm_x_quick.sh" >&2
  exit 2
fi

read -r -a X_PRESETS <<< "$SACM_X_PRESETS"
if [ "${#X_PRESETS[@]}" -eq 0 ]; then
  echo "ERROR: SACM_X_PRESETS is empty" >&2
  exit 2
fi
if [ -n "$SACM_X_GPUS" ]; then
  read -r -a X_GPUS <<< "$SACM_X_GPUS"
else
  X_GPUS=()
  for i in "${!X_PRESETS[@]}"; do
    X_GPUS+=("$((i + 1))")
  done
fi
if [ "${#X_GPUS[@]}" -ne "${#X_PRESETS[@]}" ]; then
  echo "ERROR: SACM_X_PRESETS has ${#X_PRESETS[@]} entries but " >&2
  echo "SACM_X_GPUS has ${#X_GPUS[@]}; counts must match." >&2
  exit 2
fi
for i in "${!X_GPUS[@]}"; do
  if [ "${X_GPUS[$i]}" = "0" ]; then
    echo "ERROR: GPU 0 is reserved for official SACM." >&2
    exit 2
  fi
  for j in $(seq 0 $((i - 1))); do
    if [ "${X_GPUS[$i]}" = "${X_GPUS[$j]}" ]; then
      echo "ERROR: duplicate SACM-X GPU ${X_GPUS[$i]}." >&2
      exit 2
    fi
  done
done

# Refuse to compete with an existing job on any GPU we will use.
if [ "$ALLOW_BUSY" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  for gpu in 0 "${X_GPUS[@]}"; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || used=""
    if [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -gt 2000 ]; then
      echo "ERROR: GPU $gpu already uses ${used} MiB; refusing to start." >&2
      echo "Free it, choose another server, or set ALLOW_BUSY=1." >&2
      nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null || true
      exit 3
    fi
  done
fi

mkdir -p "$OFFICIAL_DIR" "$SACM_X_DIR" "$LOG_DIR"

WANDB_ARGS=(--use_wandb "$USE_WANDB")
if [ "$USE_WANDB" = "true" ]; then
  WANDB_ARGS+=(--wandb_group "$GROUP")
fi

run_official() {
  echo "== GPU0: SACM official =="
  "$PYTHON" -m baseline_zoo.run_one \
    --model sacm_official --stage train \
    --data-root "$DATA_ROOT" --workdir "$OFFICIAL_DIR" \
    --epochs "$EPOCHS" --batch-size 1 --seed "$SEED" \
    --device cuda:0 \
    --sam-checkpoint "$SAM_CKPT" \
    "${WANDB_ARGS[@]}" \
    --wandb_name "${GROUP}-sacm_official-seed${SEED}-train" \
    --wandb_job_type train \
    --wandb_tags "sacm_official,seed${SEED},vit_l,3shot,${GROUP}"

  echo "== GPU0: SACM official eval (20 datasets) =="
  "$PYTHON" scripts/pipeline/run_baseline_test.py \
    --method sacm_official \
    --protocol "$PROTOCOL" \
    --seed "$SEED" \
    --checkpoint "$OFFICIAL_DIR/best_model.pth" \
    --pred-root "$OFFICIAL_DIR/predictions" \
    --paper-root "$OFFICIAL_DIR/paper" \
    --data-root "$DATA_ROOT" \
    --device cuda:0 \
    "${WANDB_ARGS[@]}" \
    --wandb_name "${GROUP}-sacm_official-seed${SEED}-eval" \
    --wandb_job_type eval \
    --wandb_tags "sacm_official,seed${SEED},vit_l,3shot,paper_metrics,${GROUP}" \
    --skip-visualize
}

run_sacm_x() {
  local preset="$1"
  local gpu="$2"
  echo "== GPU${gpu}: SACM-X (ViT-L, preset=${preset}) =="
  GPU="$gpu" \
  MODEL=sam_l \
  PRESET="$preset" \
  EPOCHS="$EPOCHS" \
  VAL_INTERVAL="$VAL_INTERVAL" \
  SEED="$SEED" \
  CKPT="$SAM_CKPT" \
  TRAIN_ROOT="$DATA_ROOT" \
  OUTPUT_ROOT="$SACM_X_DIR" \
  MONITOR=official \
  USE_WANDB="$USE_WANDB" \
  WANDB_MODE="$WANDB_MODE" \
  WANDB_RUN_GROUP="$GROUP" \
  SKIP_AGGREGATE=true \
  bash scripts/sacm_x/run_sacm_x_quick.sh
}

echo "seed=$SEED epochs=$EPOCHS presets=${X_PRESETS[*]} gpus=${X_GPUS[*]}"
echo "val_interval=$VAL_INTERVAL group=$GROUP"
echo "output=$ROOT"

run_official >"$LOG_DIR/sacm_official_train.log" 2>&1 &
PID0=$!

PIDS=()
for i in "${!X_PRESETS[@]}"; do
  preset="${X_PRESETS[$i]}"
  gpu="${X_GPUS[$i]}"
  run_sacm_x "$preset" "$gpu" >"$LOG_DIR/sacm_x_${preset}.log" 2>&1 &
  PIDS+=("$!")
done

cleanup() {
  kill "$PID0" "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup INT TERM

wait "$PID0"
RC0=$?

RCX=()
for pid in "${PIDS[@]}"; do
  rc=0
  wait "$pid" || rc=$?
  RCX+=("$rc")
done

AGG_RC=0
echo
echo "== aggregate SACM-X presets =="
"$PYTHON" scripts/sacm_x/aggregate_results.py \
  --root "$SACM_X_DIR" --out "$SACM_X_DIR/tables" || AGG_RC=$?

echo
echo "== status =="
echo "SACM official: $([ "$RC0" -eq 0 ] && echo OK || echo FAILED)  log=$LOG_DIR/sacm_official_train.log"
ALL_X_OK=1
for i in "${!X_PRESETS[@]}"; do
  preset="${X_PRESETS[$i]}"
  rc="${RCX[$i]}"
  [ "$rc" -eq 0 ] || ALL_X_OK=0
  echo "SACM-X ${preset}: $([ "$rc" -eq 0 ] && echo OK || echo FAILED)  log=$LOG_DIR/sacm_x_${preset}.log"
done
echo "official results: $OFFICIAL_DIR/paper"
echo "SACM-X results:   $SACM_X_DIR/tables"

if [ "$RC0" -ne 0 ] || [ "$ALL_X_OK" -ne 1 ] || [ "$AGG_RC" -ne 0 ]; then
  exit 1
fi
