#!/usr/bin/env bash
# One-click BCTD-SAM baseline-zoo run (3-shot, 20 datasets, 200 epochs).
#
# Train from scratch (default):
#   bash scripts/run_bctd_baseline_seeds03_200.sh
#
# Reproduce from existing method-tree BCTD checkpoints without retraining:
#   MODE=repro SRC_RUN_ROOT=results/runs/topostrand/bctd200 \
#     bash scripts/run_bctd_baseline_seeds03_200.sh
#
# Env:
#   SEEDS="0 1 2 3", TRAIN_GPUS=0,1,2,3, EVAL_GPUS=4,5,6,7
#   SKIP_TRAIN=0, SKIP_EVAL=0, DRY_RUN=0
#   MODE=train|repro, SRC_RUN_ROOT=..., OUT_ROOT=...
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SEEDS_STR="${SEEDS:-0 1 2 3}"
SEEDS_STR="${SEEDS_STR//,/ }"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
EVAL_GPUS="${EVAL_GPUS:-4,5,6,7}"
MODE="${MODE:-train}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
SRC_RUN_ROOT="${SRC_RUN_ROOT:-results/runs/topostrand/bctd200}"
OUT_ROOT="${OUT_ROOT:-results/runs/bctd/repro}"

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
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

read -r -a SEEDS <<< "$SEEDS_STR"
read -r -a TGPUS <<< "${TRAIN_GPUS//,/ }"
read -r -a EGPUS <<< "${EVAL_GPUS//,/ }"
if [ "${#SEEDS[@]}" -eq 0 ] || [ "${#TGPUS[@]}" -eq 0 ] || [ "${#EGPUS[@]}" -eq 0 ]; then
  echo "ERROR: empty SEEDS/TRAIN_GPUS/EVAL_GPUS" >&2
  exit 2
fi

if [ "$MODE" = repro ]; then
  if [ "$SKIP_EVAL" = 1 ]; then
    echo "MODE=repro with SKIP_EVAL=1: nothing to do"
    exit 0
  fi
  echo "reproduce BCTD baseline from $SRC_RUN_ROOT -> $OUT_ROOT"
  case "$OUT_ROOT" in
    /*) out_root="$OUT_ROOT" ;;
    *) out_root="$REPO/$OUT_ROOT" ;;
  esac
  case "$SRC_RUN_ROOT" in
    /*) src_root="$SRC_RUN_ROOT" ;;
    *) src_root="$REPO/$SRC_RUN_ROOT" ;;
  esac
  batch_size="${#EGPUS[@]}"
  if [ "$batch_size" -lt 1 ]; then batch_size=1; fi
  for ((start=0; start<${#SEEDS[@]}; start+=batch_size)); do
    pids=()
    for ((offset=0; offset<batch_size && start+offset<${#SEEDS[@]}; offset++)); do
      i=$((start + offset))
      seed="${SEEDS[$i]}"
      gpu="${EGPUS[$offset]}"
      workdir="$out_root/seed_${seed}"
      src_checkpoint="$src_root/seed_${seed}/best_model_fixed3.pth"
      checkpoint="$workdir/best_model_fixed3.pth"
      if [ ! -f "$src_checkpoint" ]; then
        echo "ERROR: missing checkpoint $src_checkpoint" >&2
        exit 1
      fi
      derive_cmd=("$PYTHON" -m baseline_zoo.models.bctd.derive_fixed3
        --input "$src_checkpoint"
        --output "$checkpoint"
        --deployment-head 3)
      cmd=("$PYTHON" -m baseline_zoo.models.bctd.evaluate
        --data-root "$REPO/data/sacm_3shot_seed${seed}"
        --protocol "$REPO/data/sacm_3shot_seed${seed}/protocol_manifest.csv"
        --workdir "$workdir"
        --checkpoint "$checkpoint"
        --seed "$seed" --device "cuda:${gpu}" --use-wandb 0
        --tta-flip --rcd 1 --as-calibration 1 --atc 0
        --tile 1024 --stride 896
        --routing-skel-ratio 1.0 --routing-area-ratio 0.7)
      if [ "$DRY_RUN" = 1 ]; then
        printf 'DRY derive seed=%s: %q ' "$seed" "${derive_cmd[0]}"
        printf '%q ' "${derive_cmd[@]:1}"; printf '\n'
        printf 'DRY repro seed=%s gpu=%s: %q ' "$seed" "$gpu" "${cmd[0]}"
        printf '%q ' "${cmd[@]:1}"; printf '\n'
        continue
      fi
      mkdir -p "$workdir" "$REPO/results/logs"
      "${derive_cmd[@]}" >> "$REPO/results/logs/bctd_baseline_seed${seed}.log" 2>&1
      "${cmd[@]}" >> "$REPO/results/logs/bctd_baseline_seed${seed}.log" 2>&1 &
      pids+=("$!")
    done
    if [ "${#pids[@]}" -gt 0 ]; then
      for pid in "${pids[@]}"; do wait "$pid"; done
    fi
  done
  echo "DONE BCTD-SAM baseline reproduction seeds=${SEEDS[*]} root=$OUT_ROOT"
  exit 0
fi

DRY_ARGS=()
if [ "$DRY_RUN" = 1 ]; then
  DRY_ARGS=(--dry-run)
fi

if [ "$SKIP_TRAIN" != 1 ]; then
  bash scripts/run_train.sh --model bctd \
    --seeds "${SEEDS[@]}" --gpus "$(IFS=,; echo "${TGPUS[*]}")" \
    "${DRY_ARGS[@]}"
fi

if [ "$SKIP_EVAL" != 1 ]; then
  bash scripts/run_eval.sh --model bctd \
    --seeds "${SEEDS[@]}" --gpus "$(IFS=,; echo "${EGPUS[*]}")" \
    "${DRY_ARGS[@]}"
fi

echo "DONE BCTD-SAM baseline seeds=${SEEDS[*]}"
