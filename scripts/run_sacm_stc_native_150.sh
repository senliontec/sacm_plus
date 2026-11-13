#!/usr/bin/env bash
# One-click native-control probe:
#   train native multi-candidate model (no specialized losses, no fusion)
#   derive the fixed candidate-3 checkpoint
#   evaluate 20 datasets
#   summarize against the existing SACM official tables
#
# Usage (server, repo root, preferably under tmux):
#   bash scripts/run_sacm_stc_native_150.sh
#   DRY_RUN=1 bash scripts/run_sacm_stc_native_150.sh
#   SEEDS="0" bash scripts/run_sacm_stc_native_150.sh
#   SEEDS="0 1 2 3" EPOCHS=150 bash scripts/run_sacm_stc_native_150.sh
#
# Env overrides:
#   SEEDS="0 1 2 3"       default four seeds
#   EPOCHS=150
#   TRAIN_GPUS=0,1,2,3    one GPU per seed; waves if fewer GPUs
#   EVAL_GPUS=0,1,2,3
#   OUT_ROOT=results/runs/sacm_stc_native_150
#   SUMMARY_ROOT=results/ablations/sacm_stc_native_150
#   OFFICIAL_ROOT=results/runs/sacm_official/default
#   EXPECTED_DATASETS=20  MAX_USED_MIB=2000  MIN_FREE_GB=30
#   SKIP_TRAIN=0 SKIP_EVAL=0 RESUME=1 FORCE_DERIVE=0 FORCE_EVAL=0
#   WANDB_MODE=online WANDB_PROJECT=sacm WANDB_ENTITY=buaazqk
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SEEDS_STR="${SEEDS:-0 1 2 3}"
SEEDS_STR="${SEEDS_STR//,/ }"
EPOCHS="${EPOCHS:-150}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
OUT_ROOT="${OUT_ROOT:-results/runs/sacm_stc_native_150}"
SUMMARY_ROOT="${SUMMARY_ROOT:-results/ablations/sacm_stc_native_150}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-results/runs/sacm_official/default}"
EXPECTED_DATASETS="${EXPECTED_DATASETS:-20}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"
WANDB_GROUP="${WANDB_GROUP:-sacm_stc_native_150}"

as_bool() {
  case "${1,,}" in
    1|true|yes|on) printf '1\n' ;;
    0|false|no|off) printf '0\n' ;;
    *) echo "ERROR: expected 0/1, got '$1'" >&2; exit 2 ;;
  esac
}
DRY_RUN=$(as_bool "${DRY_RUN:-0}")
SKIP_TRAIN=$(as_bool "${SKIP_TRAIN:-0}")
SKIP_EVAL=$(as_bool "${SKIP_EVAL:-0}")
RESUME=$(as_bool "${RESUME:-1}")
FORCE_DERIVE=$(as_bool "${FORCE_DERIVE:-0}")
FORCE_EVAL=$(as_bool "${FORCE_EVAL:-0}")
ALLOW_BUSY=$(as_bool "${ALLOW_BUSY:-0}")
ALLOW_LOW_DISK=$(as_bool "${ALLOW_LOW_DISK:-0}")

read -r -a SEED_ARGS <<< "$SEEDS_STR"
if [ "${#SEED_ARGS[@]}" -eq 0 ]; then
  echo "ERROR: SEEDS is empty" >&2
  exit 2
fi
for seed in "${SEED_ARGS[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || {
    echo "ERROR: SEEDS must be non-negative integers; got '$seed'" >&2
    exit 2
  }
done
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EPOCHS must be positive" >&2
  exit 2
fi
if ! [[ "$EXPECTED_DATASETS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EXPECTED_DATASETS must be positive" >&2
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
export TMPDIR="${TMPDIR:-$REPO/results/.tmp}"
mkdir -p "$TMPDIR"

IFS=',' read -r -a TRAIN_GPU_ARR <<< "$TRAIN_GPUS"
IFS=',' read -r -a EVAL_GPU_ARR <<< "$EVAL_GPUS"
if [ "${#TRAIN_GPU_ARR[@]}" -eq 0 ] || [ "${#EVAL_GPU_ARR[@]}" -eq 0 ]; then
  echo "ERROR: GPU lists must not be empty" >&2
  exit 2
fi

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: nvidia-smi not found; run on the GPU server" >&2
  exit 2
}
for gpu in "${TRAIN_GPU_ARR[@]}" "${EVAL_GPU_ARR[@]}"; do
  used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
  [[ "$used" =~ ^[0-9]+$ ]] || {
    echo "ERROR: GPU $gpu is not visible" >&2
    exit 2
  }
  if [ "$ALLOW_BUSY" != "1" ] && [ "$used" -gt "$MAX_USED_MIB" ]; then
    echo "ERROR: GPU $gpu already uses ${used} MiB (MAX_USED_MIB=$MAX_USED_MIB)" >&2
    echo "Free it or set ALLOW_BUSY=1." >&2
    exit 2
  fi
done
FREE_GB="$(df -Pk "$REPO" | awk 'NR==2 {print int($4/1024/1024)}')"
if [ "$ALLOW_LOW_DISK" != "1" ] && [ -n "$FREE_GB" ] \
   && [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
  echo "ERROR: only ${FREE_GB} GiB free under $REPO; need ${MIN_FREE_GB} GiB" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-$REPO/results/logs/sacm_stc_native_150_$STAMP}"
mkdir -p "$LOG_ROOT" "$OUT_ROOT" "$SUMMARY_ROOT"
LOCK_FILE="$OUT_ROOT/.run.lock"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "ERROR: another native-150 run is active" >&2; exit 2; }

for seed in "${SEED_ARGS[@]}"; do
  [ -f "data/sacm_3shot_seed$seed/protocol_manifest.csv" ] || {
    echo "ERROR: missing protocol for seed $seed" >&2
    exit 2
  }
  [ -f "$OFFICIAL_ROOT/seed_$seed/eval/tables/all_metrics.csv" ] || {
    echo "ERROR: missing official eval table for seed $seed" >&2
    exit 2
  }
done
[ -f checkpoints/sam_vit_l_0b3195.pth ] || {
  echo "ERROR: missing checkpoints/sam_vit_l_0b3195.pth" >&2
  exit 2
}
grep -q -- '--allow-native' scripts/pipeline/make_fixed3_checkpoint.py || {
  echo "ERROR: make_fixed3_checkpoint.py lacks --allow-native" >&2
  exit 2
}

source "$REPO/scripts/wandb_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
if [ "$DRY_RUN" = 0 ] \
   && [ "$WANDB_MODE" != "offline" ] \
   && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled but no credentials were found." >&2
    echo "Set WANDB_API_KEY, or use WANDB_MODE=offline/disabled." >&2
    exit 1
  fi
fi

COMMON=(
  --model sacm_stc
  --stage train
  --epochs "$EPOCHS"
  --sam-checkpoint checkpoints/sam_vit_l_0b3195.pth
  --bce-weight 0.7 --dice-weight 0.3 --bce-pos-weight 2
  --binary-targets 0 --deploy-threshold 0.5
  --stage2-loss-scope all --stage2-loss-aggregate deployment
  --stage2-aux-weight 0.25
  --topology-head all --topology-aggregate deployment
  --topology-aux-weight 0.25 --topology-resolution 512
  --topology-source native --topology-warmup-epochs 5
  --cl-dice-weight 0.1 --skeleton-recall-weight 0.1
  --specialized-head-weight 0.1 --specialized-topology-mode cldice
  --deep-sup-weight 0.3 --iou-loss-weight 0
  --head-select gating --deployment-head 2
  --head-fusion 0 --fusion-loss-weight 0.0
  --head-fusion-mode linear --head-fusion-warmup-epochs 0
  --image-dense-prompt 0 --morph-prompt 0 --polarity-augment 0
  --topology-quality 0 --use-stc 0 --use-typed-stage2 0
  --save-every 0 --num-workers 0
  --use_wandb true
  --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY"
  --wandb_group "$WANDB_GROUP"
)
# The native arm is the only training factor: no specialized losses,
# no fusion.  All candidates keep the dense and topology supervision.
TRAIN_EXTRA=(--specialized-heads 0)

echo "============================================================"
echo " SACM-STC native multi-candidate probe"
echo " seeds: ${SEED_ARGS[*]}  epochs: $EPOCHS"
echo " train gpus: ${TRAIN_GPU_ARR[*]}  eval gpus: ${EVAL_GPU_ARR[*]}"
echo " run root:   $OUT_ROOT"
echo " summary:    $SUMMARY_ROOT"
echo " log:        $LOG_ROOT"
echo "============================================================"

train_one() {
  local seed=$1 gpu=$2
  local root="$OUT_ROOT/seed_$seed"
  local log="$LOG_ROOT/train_seed_$seed.log"
  mkdir -p "$root"
  if [ "$SKIP_TRAIN" = 1 ]; then
    echo "[train:$seed] skipped (SKIP_TRAIN=1)"
    return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run][train:$seed] cuda:$gpu -> $root  ($EPOCHS epochs)"
    return 0
  fi
  if [ "$RESUME" = 1 ] && [ -f "$root/run_info.json" ] \
     && [ -f "$root/best_model.pth" ] \
     && grep -Eq '"status"[[:space:]]*:[[:space:]]*"done"' \
        "$root/run_info.json" \
     && grep -Eq "\"epochs\"[[:space:]]*:[[:space:]]*$EPOCHS([^0-9]|$)" \
        "$root/run_info.json"; then
    echo "[train:$seed] already complete; skip"
    return 0
  fi
  echo "[train:$seed] cuda:$gpu -> $root"
  local extra=()
  if [ "$RESUME" = 1 ] && [ -f "$root/run_info.json" ]; then
    extra+=(--resume)
  fi
  "$PYTHON" -m baseline_zoo.run_one \
    "${COMMON[@]}" "${TRAIN_EXTRA[@]}" \
    --data-root "data/sacm_3shot_seed$seed" \
    --workdir "$root" --seed "$seed" --device "cuda:$gpu" \
    --wandb_name "sacm_stc_native_150_seed$seed" \
    --wandb_tags "sacm_stc,native,150,probe" \
    "${extra[@]}" >>"$log" 2>&1
  echo "[train:$seed] done"
}

derive_one() {
  local seed=$1
  local run="$OUT_ROOT/seed_$seed"
  local input="$run/best_model.pth"
  local output="$run/best_model_fixed_candidate.pth"
  if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run][derive:$seed] $input -> $output"
    return 0
  fi
  if [ ! -f "$input" ]; then
    echo "ERROR: missing checkpoint $input" >&2
    exit 1
  fi
  local force=()
  [ "$FORCE_DERIVE" = 1 ] && force+=(--force)
  "$PYTHON" scripts/pipeline/make_fixed3_checkpoint.py \
    --input "$input" --output "$output" --deployment-head 3 \
    --allow-native "${force[@]}" >>"$LOG_ROOT/derive_seed_$seed.log" 2>&1
  echo "[derive:$seed] done"
}

eval_one() {
  local seed=$1 gpu=$2
  local run="$OUT_ROOT/seed_$seed"
  local ckpt="$run/best_model_fixed_candidate.pth"
  local log="$LOG_ROOT/eval_seed_$seed.log"
  local eval_root="$OUT_ROOT/eval/seed_$seed"
  if [ "$SKIP_EVAL" = 1 ]; then
    echo "[eval:$seed] skipped (SKIP_EVAL=1)"
    return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run][eval:$seed] cuda:$gpu -> $eval_root"
    return 0
  fi
  mkdir -p "$eval_root"
  echo "[eval:$seed] cuda:$gpu -> $eval_root"
  local force=()
  [ "$FORCE_EVAL" = 1 ] && force+=(--no-resume)
  "$PYTHON" scripts/pipeline/run_baseline_test.py \
    --method sacm_stc \
    --protocol "data/sacm_3shot_seed$seed/protocol_manifest.csv" \
    --seed "$seed" --variant native_fixed150 \
    --checkpoint "$ckpt" --sam-checkpoint checkpoints/sam_vit_l_0b3195.pth \
    --pred-root "$OUT_ROOT/predictions/seed_$seed" \
    --paper-root "$eval_root" --device "cuda:$gpu" \
    --metrics dice,iou,cldice,hd95 --skip-visualize \
    --use_wandb true --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" --wandb_group "${WANDB_GROUP}_eval" \
    --wandb_name "native_fixed150_seed$seed" \
    --wandb_tags "sacm_stc,native,150,eval" \
    --allow-stale-checkpoint "${force[@]}" \
    >>"$log" 2>&1
  echo "[eval:$seed] done"
}

run_waves() {
  local fn=$1; shift
  local gpus=("$@")
  local i=0
  while [ "$i" -lt "${#SEED_ARGS[@]}" ]; do
    local pids=()
    local wave_seeds=()
    local j=0
    while [ "$j" -lt "${#gpus[@]}" ] && [ "$i" -lt "${#SEED_ARGS[@]}" ]; do
      wave_seeds+=("${SEED_ARGS[$i]}")
      "$fn" "${SEED_ARGS[$i]}" "${gpus[$j]}" &
      pids+=("$!")
      i=$((i + 1))
      j=$((j + 1))
    done
    local k
    for k in "${!pids[@]}"; do
      wait "${pids[$k]}"
    done
    echo "  wave done: ${wave_seeds[*]}"
  done
}

if [ "$SKIP_TRAIN" != 1 ]; then
  echo "[1/4] training native arms"
  run_waves train_one "${TRAIN_GPU_ARR[@]}"
else
  echo "[1/4] training skipped"
fi

echo "[2/4] deriving fixed candidate-3 checkpoints"
for seed in "${SEED_ARGS[@]}"; do
  derive_one "$seed"
done

echo "[3/4] evaluating 20 datasets"
if [ "$SKIP_EVAL" != 1 ]; then
  run_waves eval_one "${EVAL_GPU_ARR[@]}"
else
  echo "[3/4] evaluation skipped"
fi

echo "[4/4] summarizing against official tables"
if [ "$DRY_RUN" = 1 ]; then
  echo "[dry-run][summary] skipped"
elif [ "$SKIP_EVAL" = 1 ]; then
  echo "[4/4] summary skipped (SKIP_EVAL=1)"
else
  "$PYTHON" scripts/pipeline/summarize_fixed3_seeds.py \
    --fixed3-root "$OUT_ROOT" \
    --official-root "$OFFICIAL_ROOT" \
    --seeds "${SEED_ARGS[@]}" \
    --epochs "$EPOCHS" \
    --expected-datasets "$EXPECTED_DATASETS" \
    --fixed3-checkpoint-name best_model_fixed_candidate.pth \
    --output-root "$SUMMARY_ROOT"
  echo "summary: $SUMMARY_ROOT/summary.md"
fi
echo "done: $OUT_ROOT"
