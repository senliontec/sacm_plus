#!/usr/bin/env bash
# One-click SPA-SAM baseline run (independent baseline_zoo model):
#   train  seeds 0-3 at 800 epochs with structure-preserving augmentation
#   derive fixed candidate-3 checkpoints
#   evaluate 20 datasets for every seed
#   write a four-seed summary and compare against DCD/SACM when available
#
# Default GPU placement:
#   training:    GPUs 0-3
#   evaluation:  GPUs 4-7  (after training; the two stages do not overlap)
#
# Usage:
#   bash scripts/run_spa_sam_seeds03_800.sh
#   DRY_RUN=1 bash scripts/run_spa_sam_seeds03_800.sh
#   EPOCHS=200 bash scripts/run_spa_sam_seeds03_800.sh
#
# Env:
#   SEEDS, EPOCHS, TRAIN_GPUS, EVAL_GPUS, TRAIN_AUG
#   RUN_ROOT=results/runs/spa_sam/default
#   SUMMARY_ROOT=results/ablations/spa_sam_800
#   DCD_ROOT (optional: existing DCD 4-seed eval root)
#   SACM_ROOT (optional: existing SACM 4-seed eval root)
#   RESUME=1 SKIP_TRAIN=0 SKIP_EVAL=0 DERIVE_ONLY=0
#   FORCE_DERIVE=0 FORCE_EVAL=0 DRY_RUN=0
#   USE_WANDB=1 WANDB_MODE=online WANDB_PROJECT=sacm WANDB_ENTITY=buaazqk
#   WANDB_GROUP=spa_sam_800  WANDB_PREFIX=spa-sam
#   EXTRA_ARGS=""  (forwarded to training; used by the ablation driver)
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SEEDS_STR="${SEEDS:-0 1 2 3}"
SEEDS_STR="${SEEDS_STR//,/ }"
EPOCHS="${EPOCHS:-800}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
EVAL_GPUS="${EVAL_GPUS:-4,5,6,7}"
TRAIN_AUG="${TRAIN_AUG:-mild}"
RUN_ROOT="${RUN_ROOT:-results/runs/spa_sam/default}"
SUMMARY_ROOT="${SUMMARY_ROOT:-results/ablations/spa_sam_800}"
DCD_ROOT="${DCD_ROOT:-results/runs/sacm_stc_fixed3_800}"
SACM_ROOT="${SACM_ROOT:-results/runs/sacm/default}"
WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"
WANDB_GROUP="${WANDB_GROUP:-spa_sam_800}"
WANDB_PREFIX="${WANDB_PREFIX:-spa-sam}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

as_bool() {
  case "${1,,}" in
    1|true|yes|on) printf '1\n' ;;
    0|false|no|off) printf '0\n' ;;
    *) echo "ERROR: expected 0/1, got '$1'" >&2; exit 2 ;;
  esac
}
RESUME=$(as_bool "${RESUME:-1}")
SKIP_TRAIN=$(as_bool "${SKIP_TRAIN:-0}")
SKIP_EVAL=$(as_bool "${SKIP_EVAL:-0}")
DERIVE_ONLY=$(as_bool "${DERIVE_ONLY:-0}")
FORCE_DERIVE=$(as_bool "${FORCE_DERIVE:-0}")
FORCE_EVAL=$(as_bool "${FORCE_EVAL:-0}")
DRY_RUN=$(as_bool "${DRY_RUN:-0}")
USE_WANDB=$(as_bool "${USE_WANDB:-1}")

read -r -a SEED_ARGS <<< "$SEEDS_STR"
read -r -a TRAIN_GPU_ARGS <<< "${TRAIN_GPUS//,/ }"
read -r -a EVAL_GPU_ARGS <<< "${EVAL_GPUS//,/ }"
EXTRA_ARRAY=()
if [ -n "$EXTRA_ARGS" ]; then
  read -r -a EXTRA_ARRAY <<< "$EXTRA_ARGS"
fi
if [ "${#SEED_ARGS[@]}" -eq 0 ]; then
  echo "ERROR: SEEDS is empty" >&2
  exit 2
fi
if [ "${#TRAIN_GPU_ARGS[@]}" -eq 0 ] || [ "${#EVAL_GPU_ARGS[@]}" -eq 0 ]; then
  echo "ERROR: TRAIN_GPUS/EVAL_GPUS are empty" >&2
  exit 2
fi
for seed in "${SEED_ARGS[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEEDS must be non-negative integers; got '$seed'" >&2
    exit 2
  fi
done
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EPOCHS must be positive; got '$EPOCHS'" >&2
  exit 2
fi

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  elif [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi
export PYTHON
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-$REPO/results/.tmp}"
mkdir -p "$TMPDIR"

SAM_CHECKPOINT="$REPO/checkpoints/sam_vit_l_0b3195.pth"
if [ ! -f "$SAM_CHECKPOINT" ]; then
  echo "ERROR: missing SAM checkpoint: $SAM_CHECKPOINT" >&2
  exit 1
fi
for seed in "${SEED_ARGS[@]}"; do
  data_root="$REPO/data/sacm_3shot_seed${seed}"
  if [ ! -f "$data_root/protocol_manifest.csv" ]; then
    echo "ERROR: missing protocol: $data_root/protocol_manifest.csv" >&2
    exit 1
  fi
done

source "$REPO/scripts/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
if [ "$DRY_RUN" = 0 ] && [ "$USE_WANDB" = 1 ] \
   && [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B enabled but no credentials found; set WANDB_API_KEY," \
         "run 'wandb login', or set USE_WANDB=0 / WANDB_MODE=offline." >&2
    exit 1
  fi
fi

LOCK_DIR="$REPO/results/runs/.locks"
mkdir -p "$LOCK_DIR"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_DIR/spa_sam_driver.lock"
  if ! flock -n 9; then
    echo "ERROR: another SPA-SAM driver is already running" >&2
    exit 3
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$REPO/results/logs/spa_sam_${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1

echo "============================================================"
echo " SPA-SAM | seeds ${SEEDS_STR} | ${EPOCHS} epochs | aug=${TRAIN_AUG}"
echo " train GPUs: ${TRAIN_GPUS}   eval GPUs: ${EVAL_GPUS}"
echo " run root:   ${RUN_ROOT}"
echo " summary:    ${SUMMARY_ROOT}"
echo " repo:       ${REPO}"
echo " log:        ${LOG_DIR}"
if [ -n "$EXTRA_ARGS" ]; then echo " extra args: ${EXTRA_ARGS}"; fi
if [ "$DRY_RUN" = 1 ]; then echo " mode:       DRY RUN"; fi
echo "============================================================"

if [ "$DRY_RUN" = 0 ]; then
  "$PYTHON" - "$TRAIN_GPUS" "$EVAL_GPUS" <<'PY'
import subprocess
import sys

for label, text in (('TRAIN_GPUS', sys.argv[1]), ('EVAL_GPUS', sys.argv[2])):
    for gpu in [x for x in text.replace(',', ' ').split() if x]:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free',
             '--format=csv,noheader,nounits', '-i', gpu],
            capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f'{label}: cannot query GPU {gpu}: {out.stderr}')
        free = int(out.stdout.strip().split(',')[1])
        if free < 4000:
            raise SystemExit(
                f'{label}: GPU {gpu} has only {free} MiB free; stop other jobs')
PY
fi

if [ "$DERIVE_ONLY" = 0 ] && [ "$SKIP_TRAIN" = 0 ]; then
  echo "[stage] training"
  TRAIN_PIDS=()
  for i in "${!SEED_ARGS[@]}"; do
    seed="${SEED_ARGS[$i]}"
    gpu="${TRAIN_GPU_ARGS[$((i % ${#TRAIN_GPU_ARGS[@]}))]}"
    workdir="$REPO/$RUN_ROOT/seed_${seed}"
    log="$LOG_DIR/train_seed${seed}.log"
    cmd=("$PYTHON" -m baseline_zoo.run_one --model spa_sam --stage train
      --data-root "$REPO/data/sacm_3shot_seed${seed}"
      --workdir "$workdir" --epochs "$EPOCHS" --seed "$seed"
      --device "cuda:${gpu}" --use_wandb "$USE_WANDB"
      --sam-checkpoint "$SAM_CHECKPOINT"
      --bce-weight 0.7 --dice-weight 0.3 --bce-pos-weight 2
      --binary-targets 0 --deploy-threshold 0.5
      --stage2-loss-scope all --topology-head all
      --cl-dice-weight 0.1 --skeleton-recall-weight 0.1
      --specialized-heads 1 --specialized-head-weight 0.1
      --specialized-topology-weight 0.1
      --head-fusion 1 --fusion-loss-weight 1.0
      --topology-resolution 512 --topology-warmup-epochs 5
      --deep-sup-weight 0.3 --head-select gating --deployment-head 3
      --adapter-dim-ratio 0.1 --num-workers 0 --save-every 0
      --train-aug "$TRAIN_AUG"
      --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY"
      --wandb_group "$WANDB_GROUP"
      --wandb_name "${WANDB_PREFIX}-seed${seed}")
    if [ "$RESUME" = 1 ]; then cmd+=(--resume); fi
    if [ "${#EXTRA_ARRAY[@]}" -gt 0 ]; then cmd+=("${EXTRA_ARRAY[@]}"); fi
    printf '+ '; printf '%q ' "${cmd[@]}"; printf '\n'
    if [ "$DRY_RUN" = 0 ]; then
      "${cmd[@]}" >"$log" 2>&1 &
      TRAIN_PIDS+=("$!")
      echo "  seed ${seed} -> cuda:${gpu} (pid $!) log ${log}"
    fi
  done
  if [ "$DRY_RUN" = 0 ]; then
    status=0
    for pid in "${TRAIN_PIDS[@]}"; do wait "$pid" || status=1; done
    if [ "$status" -ne 0 ]; then
      echo "ERROR: at least one training seed failed; see ${LOG_DIR}" >&2
      exit 1
    fi
  fi
fi

echo "[stage] fixed3 derivation"
for i in "${!SEED_ARGS[@]}"; do
  seed="${SEED_ARGS[$i]}"
  workdir="$REPO/$RUN_ROOT/seed_${seed}"
  input="$workdir/best_model.pth"
  output="$workdir/best_model_fixed3.pth"
  cmd=("$PYTHON" -m baseline_zoo.models.spa_sam.derive_fixed3
    --input "$input" --output "$output" --deployment-head 3)
  if [ "$FORCE_DERIVE" = 1 ]; then cmd+=(--force); fi
  printf '+ '; printf '%q ' "${cmd[@]}"; printf '\n'
  if [ "$DRY_RUN" = 0 ]; then "${cmd[@]}"; fi
done

if [ "$SKIP_EVAL" = 0 ]; then
  echo "[stage] 20-dataset evaluation"
  EVAL_PIDS=()
  for i in "${!SEED_ARGS[@]}"; do
    seed="${SEED_ARGS[$i]}"
    gpu="${EVAL_GPU_ARGS[$((i % ${#EVAL_GPU_ARGS[@]}))]}"
    workdir="$REPO/$RUN_ROOT/seed_${seed}"
    log="$LOG_DIR/eval_seed${seed}.log"
    cmd=("$PYTHON" scripts/pipeline/run_baseline_test.py
      --method spa_sam
      --protocol "$REPO/data/sacm_3shot_seed${seed}/protocol_manifest.csv"
      --seed "$seed"
      --checkpoint "$workdir/best_model_fixed3.pth"
      --pred-root "$workdir/predictions"
      --paper-root "$workdir/eval"
      --device "cuda:${gpu}"
      --use_wandb "$USE_WANDB"
      --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY"
      --wandb_group "${WANDB_GROUP}_eval"
      --wandb_name "${WANDB_PREFIX}-eval-seed${seed}"
      --skip-visualize)
    if [ "$FORCE_EVAL" = 1 ]; then cmd+=(--no-resume); fi
    printf '+ '; printf '%q ' "${cmd[@]}"; printf '\n'
    if [ "$DRY_RUN" = 0 ]; then
      "${cmd[@]}" >"$log" 2>&1 &
      EVAL_PIDS+=("$!")
      echo "  eval seed ${seed} -> cuda:${gpu} (pid $!) log ${log}"
    fi
  done
  if [ "$DRY_RUN" = 0 ]; then
    status=0
    for pid in "${EVAL_PIDS[@]}"; do wait "$pid" || status=1; done
    if [ "$status" -ne 0 ]; then
      echo "ERROR: at least one evaluation failed; see ${LOG_DIR}" >&2
      exit 1
    fi
  fi
fi

echo "[stage] summary"
summary_cmd=("$PYTHON" scripts/pipeline/summarize_spa_sam.py
  --root "$REPO/$RUN_ROOT" --out "$REPO/$SUMMARY_ROOT")
if [ -n "$DCD_ROOT" ] && [ -d "$REPO/$DCD_ROOT" ]; then
  summary_cmd+=(--dcd-root "$REPO/$DCD_ROOT")
fi
if [ -n "$SACM_ROOT" ] && [ -d "$REPO/$SACM_ROOT" ]; then
  summary_cmd+=(--sacm-root "$REPO/$SACM_ROOT")
fi
printf '+ '; printf '%q ' "${summary_cmd[@]}"; printf '\n'
if [ "$DRY_RUN" = 0 ]; then "${summary_cmd[@]}"; fi

echo "DONE  logs: ${LOG_DIR}  summary: ${SUMMARY_ROOT}/summary.md"
