#!/usr/bin/env bash
# One-click CoSNA: initialize from BCTD champion checkpoints, train the TCSU
# head for 20 epochs, and evaluate the 20-dataset protocol.
#
# Usage:
#   bash scripts/run_cosna_seeds03.sh
#   DRY_RUN=1 bash scripts/run_cosna_seeds03.sh
#
# Env:
#   SEEDS="0 1 2 3", GPUS="0,1,2,3", EPOCHS_ADD=20
#   BCTD_ROOT=results/runs/topostrand/bctd200
#   SEED0_CKPT=results/runs/topostrand/pilot200_bnd005_h2/seed_0/best_model_fixed3.pth
#   RUN_ROOT=results/runs/cosna
#   FORCE=0 (reinitialize copied checkpoints), SKIP_TRAIN=0, SKIP_EVAL=0
#   USE_WANDB=0, WANDB_PROJECT=sacm, WANDB_ENTITY=buaazqk
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SEEDS_STR="${SEEDS:-0 1 2 3}"
SEEDS_STR="${SEEDS_STR//,/ }"
GPU_STR="${GPUS:-0,1,2,3}"
BCTD_ROOT="${BCTD_ROOT:-results/runs/topostrand/bctd200}"
SEED0_CKPT="${SEED0_CKPT:-results/runs/topostrand/pilot200_bnd005_h2/seed_0/best_model_fixed3.pth}"
RUN_ROOT="${RUN_ROOT:-results/runs/cosna}"
EPOCHS_ADD="${EPOCHS_ADD:-20}"
FORCE="${FORCE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-$REPO/results/.tmp}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "$TMPDIR" "$REPO/results/logs"

if [ "$USE_WANDB" != 0 ]; then
  source "$REPO/scripts/wandb_env.sh"
fi

read -r -a SEEDS <<< "$SEEDS_STR"
read -r -a GPUS <<< "${GPU_STR//,/ }"
if [ "${#SEEDS[@]}" -eq 0 ] || [ "${#GPUS[@]}" -eq 0 ]; then
  echo "ERROR: empty SEEDS/GPUS" >&2
  exit 2
fi

run_one_seed() {
  local seed="$1" gpu="$2"
  local work="$REPO/$RUN_ROOT/seed_${seed}"
  local log="$REPO/results/logs/cosna_seed${seed}.log"
  local source
  if [ "$seed" = 0 ]; then
    source="$REPO/$SEED0_CKPT"
  else
    source="$REPO/$BCTD_ROOT/seed_${seed}/best_model_fixed3.pth"
  fi
  mkdir -p "$work"
  : > "$log"

  local epochs
  if [ ! -f "$work/best_model.pth" ] || [ "$FORCE" = 1 ]; then
    if [ ! -f "$source" ]; then
      echo "missing source checkpoint: $source" >&2
      return 2
    fi
    epochs=$("$PYTHON" -m methods.cosna.prepare_init \
      --source "$source" --workdir "$work" \
      --data-root "$REPO/data/sacm_3shot_seed${seed}" \
      --epochs-add "$EPOCHS_ADD" --force)
  else
    epochs=$("$PYTHON" - "$work/best_model.pth" "$EPOCHS_ADD" <<'PY'
import sys, torch
p, add = sys.argv[1], int(sys.argv[2])
d = torch.load(p, map_location='cpu', weights_only=False)
print(int(d.get('epoch', 0)) + add)
PY
)
  fi
  echo "seed=$seed gpu=$gpu epochs=$epochs work=$work" >> "$log"

  if [ "$SKIP_TRAIN" != 1 ]; then
    local train_cmd=("$PYTHON" -m methods.cosna.run_one --stage train
      --data-root "$REPO/data/sacm_3shot_seed${seed}" --workdir "$work"
      --epochs "$epochs" --seed "$seed" --device "cuda:${gpu}"
      --use_wandb "$USE_WANDB" --resume --resume-checkpoint best
      --lr 1e-3 --cosna-weight 1.0 --cosna-cl 0.1
      --cosna-render 1024 --cosna-coarse 512 --cosna-feature-dim 16
      --cosna-gate 0.10 --cosna-near-radius 4
      --rcd 1 --as-calibration 1 --close-radius 2 --tile 1024 --stride 896
      --routing-skel-ratio 1.0 --routing-area-ratio 0.7 --routing-mode auto
      --num-workers 0 --save-every 0
      --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY"
      --wandb_group cosna --wandb_name "cosna-seed${seed}"
      --cosna-save-last "$work/cosna_final.pth")
    if [ "$DRY_RUN" = 1 ]; then
      printf 'DRY train seed=%s gpu=%s: ' "$seed" "$gpu"; printf '%q ' "${train_cmd[@]}"; printf '\n'
    else
      "${train_cmd[@]}" >> "$log" 2>&1
    fi
  fi

  if [ "$SKIP_EVAL" != 1 ]; then
    local eval_cmd=("$PYTHON" -m methods.cosna.evaluate
      --data-root "$REPO/data/sacm_3shot_seed${seed}"
      --protocol "$REPO/data/sacm_3shot_seed${seed}/protocol_manifest.csv"
      --workdir "$work" --checkpoint "$work/best_model.pth"
      --device "cuda:${gpu}" --out "$work/eval_summary.json"
      --rcd 1 --as-calibration 1 --close-radius 2
      --tile 1024 --stride 896 --routing-skel-ratio 1.0
      --routing-area-ratio 0.7 --routing-mode auto
      --cosna-gate 0.10 --cosna-near-radius 4)
    if [ "$DRY_RUN" = 1 ]; then
      printf 'DRY eval seed=%s gpu=%s: ' "$seed" "$gpu"; printf '%q ' "${eval_cmd[@]}"; printf '\n'
    else
      "${eval_cmd[@]}" >> "$log" 2>&1
    fi
  fi
  echo "DONE seed=$seed" >> "$log"
}

pids=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  run_one_seed "$seed" "$gpu" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
