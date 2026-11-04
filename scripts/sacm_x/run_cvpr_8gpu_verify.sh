#!/usr/bin/env bash
# One-click 8-GPU verification screen for the current CVPR hypothesis.
#
# This script deliberately writes to a NEW screen root so it cannot overwrite
# the old results in results/runs/cvpr8gpu_screen (including its eval/ trees).
#
# Allocation:
#   GPU 0-4: full_fixed2_aux        seed 0-4   (5-seed baseline)
#   GPU 5:   full_fixed2_scnp       seed 0     (prevention)
#   GPU 6:   full_fixed2_resmoe     seed 0     (learned-gate repair)
#   GPU 7:   full_fixed2_resmoe_dg  seed 0     (disagreement-gate fallback)
#
# GPU 7 can be switched to the encoder hybrid control:
#   GPU7_PRESET=full_fixed2_lora_evp bash scripts/sacm_x/run_cvpr_8gpu_verify.sh
#
# This is train+validation only.  It does NOT open the 20 test sets; after the
# screen, evaluate only the winning candidate with eval_sacm_x_20.sh.
#
# Usage:
#   bash scripts/sacm_x/run_cvpr_8gpu_verify.sh
#   ALLOW_DIRTY=1 bash scripts/sacm_x/run_cvpr_8gpu_verify.sh
#   WAIT_FOR_GPUS=1 bash scripts/sacm_x/run_cvpr_8gpu_verify.sh
#
# Outputs:
#   $SCREEN_ROOT/sacm_x/<preset>/seed_<seed>/
#     best_model.pth, last_model.pth, checkpoints/
#     metrics_history.json, train_config.json, run_info.json, train.log
#     provenance/{git.json,code.patch,untracked.json,source_manifest.json,
#                 code_version.json}
#     wandb_run.json
#   $SCREEN_ROOT/code_version.txt
#   $SCREEN_ROOT/code.patch
#   $SCREEN_ROOT/untracked_files.txt
#   $SCREEN_ROOT/jobs.csv
#   results/ablation_logs/cvpr8gpu_verify_<preset>_seed<seed>_<stamp>.log
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PY=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PY=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PY=python
  fi
fi
export PYTHON="$PY"

EPOCHS="${EPOCHS:-300}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
SCREEN_ROOT="${SCREEN_ROOT:-results/runs/cvpr8gpu_verify}"
LOG_DIR="${LOG_DIR:-results/ablation_logs}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-7200}"
GPU_USED_LIMIT="${GPU_USED_LIMIT:-2000}"
DRY_RUN="${DRY_RUN:-0}"
GPU7_PRESET="${GPU7_PRESET:-full_fixed2_resmoe_dg}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

case "$GPU7_PRESET" in
  full_fixed2_resmoe_dg|full_fixed2_lora_evp) ;;
  *)
    echo "ERROR: GPU7_PRESET must be full_fixed2_resmoe_dg or full_fixed2_lora_evp" >&2
    exit 2
    ;;
esac

# Preserve the already-running old screen and its eval/ outputs.
case "$SCREEN_ROOT" in
  results/runs/cvpr8gpu_screen|results/runs/cvpr8gpu_screen/|*/cvpr8gpu_screen)
    echo "ERROR: refusing to write into the old screen root: $SCREEN_ROOT" >&2
    echo "Choose a new root, e.g. SCREEN_ROOT=results/runs/cvpr8gpu_verify_2" >&2
    exit 2
    ;;
esac

# gpu preset seed
JOB_SPECS=(
  "0 full_fixed2_aux 0"
  "1 full_fixed2_aux 1"
  "2 full_fixed2_aux 2"
  "3 full_fixed2_aux 3"
  "4 full_fixed2_aux 4"
  "5 full_fixed2_scnp 0"
  "6 full_fixed2_resmoe 0"
  "7 ${GPU7_PRESET} 0"
)

# Fail before training if a required wrapper, preset, dataset or base
# checkpoint is missing.  This prevents a screen that silently skips a job.
missing=()
for f in \
  src/models/sam_network/residual_moe.py \
  src/models/sacm_x/configs.py \
  src/core/trainer.py \
  src/train/trainer.py \
  scripts/sacm_x/run_full_fixed2_aux.sh \
  scripts/sacm_x/run_full_fixed2_scnp.sh \
  scripts/sacm_x/run_full_fixed2_resmoe.sh \
  scripts/sacm_x/run_full_fixed2_resmoe_dg.sh \
  scripts/sacm_x/run_full_fixed2_lora_evp.sh \
  "$CKPT"; do
  [ -f "$f" ] || missing+=("$f")
done
for p in full_fixed2_aux full_fixed2_scnp full_fixed2_resmoe \
         full_fixed2_resmoe_dg; do
  grep -q "\"$p\"" src/models/sacm_x/configs.py || missing+=("preset:$p")
done
for s in 0 1 2 3 4; do
  [ -f "data/sacm_3shot_seed${s}/protocol_manifest.csv" ] \
    || missing+=("data/sacm_3shot_seed${s}/protocol_manifest.csv")
done
if [ "${#missing[@]}" -gt 0 ]; then
  printf 'ERROR: missing prerequisites:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 2
fi

# Code provenance: a dirty tree is allowed only explicitly, but the exact
# diff/untracked snapshot is written to the screen root either way.
CODE_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
CODE_STATUS="$(git status --porcelain 2>/dev/null || true)"
if [ "$CODE_COMMIT" = "unknown" ]; then
  echo "ERROR: git commit unknown; run from a git checkout." >&2
  exit 2
fi
if [ -n "$CODE_STATUS" ] && [ "$ALLOW_DIRTY" != "1" ]; then
  echo "ERROR: working tree is dirty; commit/tag the exact code first." >&2
  echo "       ALLOW_DIRTY=1 writes a code.patch/untracked snapshot and continues." >&2
  git status --porcelain >&2 || true
  exit 2
fi
if [ -n "$CODE_STATUS" ]; then
  echo "WARNING: ALLOW_DIRTY=1; saving a code snapshot for this screen." >&2
fi

# Do not overwrite a previous run with the same preset/seed.  Use a new
# SCREEN_ROOT for a fresh screen; only set ALLOW_EXISTING=1 intentionally.
existing=()
for spec in "${JOB_SPECS[@]}"; do
  read -r gpu preset seed <<< "$spec"
  run_root="$SCREEN_ROOT/sacm_x/$preset/seed_$seed"
  if [ -d "$run_root" ] \
      && [ -n "$(find "$run_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    existing+=("$run_root")
  fi
done
if [ "${#existing[@]}" -gt 0 ] && [ "$ALLOW_EXISTING" != "1" ]; then
  echo "ERROR: screen root already contains run outputs:" >&2
  printf '  %s\n' "${existing[@]}" >&2
  echo "Choose a new SCREEN_ROOT or set ALLOW_EXISTING=1 intentionally." >&2
  exit 2
fi

# Wait for the old 20-dataset eval to finish without killing it.  This is the
# normal path after the previous screen; GPUs are released one by one.
gpu_busy() {
  local gpu used busy=()
  for gpu in 0 1 2 3 4 5 6 7; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    if [[ "${used:-0}" =~ ^[0-9]+$ ]] && [ "$used" -gt "$GPU_USED_LIMIT" ]; then
      busy+=("$gpu:${used}MiB")
    fi
  done
  if [ "${#busy[@]}" -eq 0 ]; then
    return 1
  fi
  printf 'GPU busy: %s\n' "${busy[*]}" >&2
  return 0
}

if [ "$DRY_RUN" != "1" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found" >&2
    exit 2
  fi
  if [ "$(nvidia-smi -L | wc -l)" -lt 8 ]; then
    echo "ERROR: fewer than 8 GPUs visible" >&2
    exit 2
  fi
  if gpu_busy; then
    if [ "$WAIT_FOR_GPUS" != "1" ]; then
      echo "ERROR: GPUs are busy; set WAIT_FOR_GPUS=1 to wait." >&2
      exit 2
    fi
    wait_start="$(date +%s)"
    while gpu_busy; do
      elapsed=$(( $(date +%s) - wait_start ))
      if [ "$elapsed" -ge "$GPU_WAIT_SECONDS" ]; then
        echo "ERROR: GPUs still busy after ${GPU_WAIT_SECONDS}s." >&2
        exit 2
      fi
      echo "waiting for all 8 GPUs (${elapsed}s/${GPU_WAIT_SECONDS}s)..." >&2
      sleep 60
    done
  fi
fi

mkdir -p "$SCREEN_ROOT"
{
  echo "commit=$CODE_COMMIT"
  echo "dirty=$([ -n "$CODE_STATUS" ] && echo true || echo false)"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "epochs=$EPOCHS"
  echo "val_interval=$VAL_INTERVAL"
  echo "screen_root=$SCREEN_ROOT"
  echo "gpu7_preset=$GPU7_PRESET"
  echo "use_wandb=$USE_WANDB"
  echo "wandb_mode=$WANDB_MODE"
} > "$SCREEN_ROOT/code_version.txt"
if [ -n "$CODE_STATUS" ]; then
  git diff --no-ext-diff HEAD > "$SCREEN_ROOT/code.patch" 2>/dev/null || true
  git ls-files --others --exclude-standard \
    > "$SCREEN_ROOT/untracked_files.txt" 2>/dev/null || true
fi

echo "============================================================"
echo " CVPR 8-GPU verification screen (new root)"
echo " root:     $SCREEN_ROOT"
echo " epochs:   $EPOCHS (train+val only; test sets untouched)"
echo " wandb:    $USE_WANDB ($WANDB_MODE)"
echo " python:   $PY"
echo " git:      $(printf '%s' "$CODE_COMMIT" | cut -c1-12) (dirty=$([ -n "$CODE_STATUS" ] && echo true || echo false))"
echo " gpu7:     $GPU7_PRESET"
echo "============================================================"
for spec in "${JOB_SPECS[@]}"; do
  read -r gpu preset seed <<< "$spec"
  printf 'GPU %s  ->  %-28s seed %s\n' "$gpu" "$preset" "$seed"
done
echo "============================================================"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1: preflight only; no training launched."
  exit 0
fi

PIDS=()
GPUS=()
PRESETS=()
SEEDS=()
LOGS=()
RUN_ROOTS=()
for spec in "${JOB_SPECS[@]}"; do
  read -r gpu preset seed <<< "$spec"
  log="$LOG_DIR/cvpr8gpu_verify_${preset}_seed${seed}_${STAMP}.log"
  run_root="$SCREEN_ROOT/sacm_x/$preset/seed_$seed"
  echo "LAUNCH GPU $gpu -> $preset / seed $seed"
  PYTHON="$PY" GPU="$gpu" SEED="$seed" EPOCHS="$EPOCHS" \
  VAL_INTERVAL="$VAL_INTERVAL" CKPT="$CKPT" \
  TRAIN_ROOT="data/sacm_3shot_seed${seed}" \
  CANONICAL_ROOT="$SCREEN_ROOT" OUTPUT_ROOT= SCNP=false \
  USE_WANDB="$USE_WANDB" WANDB_MODE="$WANDB_MODE" \
  NUM_WORKERS="$NUM_WORKERS" \
  WANDB_NAME="sacm_x_cvpr8v2_${preset}_seed${seed}" \
    bash "scripts/sacm_x/run_${preset}.sh" train \
    > "$log" 2>&1 &
  PIDS+=("$!")
  GPUS+=("$gpu")
  PRESETS+=("$preset")
  SEEDS+=("$seed")
  LOGS+=("$log")
  RUN_ROOTS+=("$run_root")
done

RC=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "DONE  ${PRESETS[$i]}/seed${SEEDS[$i]}  (${LOGS[$i]})"
  else
    echo "FAIL  ${PRESETS[$i]}/seed${SEEDS[$i]}  (${LOGS[$i]})"
    tail -n 40 "${LOGS[$i]}" || true
    RC=1
  fi
done

# Verify the data that matters for the next decision.  Missing artifacts
# make the run unusable for the paper, so report them explicitly.
JOBS_CSV="$SCREEN_ROOT/jobs.csv"
printf 'gpu,preset,seed,run_root,log,status,missing_artifacts\n' > "$JOBS_CSV"
artifact_rc=0
for i in "${!PIDS[@]}"; do
  root="${RUN_ROOTS[$i]}"
  status="done"
  [ -f "$root/best_model.pth" ] || status="missing_best_model"
  missing_artifacts=()
  for f in \
    best_model.pth \
    last_model.pth \
    metrics_history.json \
    train_config.json \
    run_info.json \
    train.log \
    provenance/git.json \
    provenance/code_version.json; do
    [ -f "$root/$f" ] || missing_artifacts+=("$f")
  done
  if [ "${#missing_artifacts[@]}" -gt 0 ]; then
    status="missing_artifacts"
    artifact_rc=1
  fi
  miss="$(IFS=';'; echo "${missing_artifacts[*]:-}")"
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "${GPUS[$i]}" "${PRESETS[$i]}" "${SEEDS[$i]}" "$root" \
    "${LOGS[$i]}" "$status" "$miss" >> "$JOBS_CSV"
  if [ "$status" = "done" ]; then
    echo "ARTIFACTS OK  ${PRESETS[$i]}/seed${SEEDS[$i]}"
  else
    echo "ARTIFACTS FAIL ${PRESETS[$i]}/seed${SEEDS[$i]}: ${miss:-see log}"
  fi
done
[ "$artifact_rc" -eq 0 ] || RC=1

END_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
END_STATUS="$(git status --porcelain 2>/dev/null || true)"
{
  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_code=$RC"
  echo "end_commit=$END_COMMIT"
  echo "end_dirty=$([ -n "$END_STATUS" ] && echo true || echo false)"
} >> "$SCREEN_ROOT/code_version.txt"
if [ "$END_COMMIT" != "$CODE_COMMIT" ]; then
  echo "WARNING: HEAD changed during the screen." >&2
fi
if [ -z "$CODE_STATUS" ] && [ -n "$END_STATUS" ]; then
  echo "WARNING: working tree became dirty during the screen." >&2
fi

echo
echo "============================================================"
echo " Verification screen finished (rc=$RC)"
echo " Run roots:"
for i in "${!RUN_ROOTS[@]}"; do
  echo "   ${PRESETS[$i]}/seed${SEEDS[$i]}: ${RUN_ROOTS[$i]}"
done
echo " Checkpoints:  <run>/best_model.pth"
echo " Histories:    <run>/metrics_history.json"
echo " Configs:      <run>/train_config.json"
echo " Provenance:   <run>/provenance/"
echo " Logs:         $LOG_DIR/cvpr8gpu_verify_*_${STAMP}.log"
echo " Screen index: $JOBS_CSV"
echo " Code state:   $SCREEN_ROOT/code_version.txt"
if [ "$RC" -eq 0 ]; then
  echo
  echo "No 20-test eval was run.  Evaluate only the winning candidate later:"
  echo "  GPU=<free> SEED=0 PRESET=<winner> \\"
  echo "  TRAIN_OUT=$SCREEN_ROOT/sacm_x/<winner>/seed_0 \\"
  echo "  bash scripts/sacm_x/eval_sacm_x_20.sh"
fi
echo "============================================================"
exit "$RC"
