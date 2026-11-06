#!/usr/bin/env bash
# Train one Skein preset into the canonical results/runs layout.
#
# The default run root is:
#   results/runs/skein/<preset>/seed<seed>/
# Training writes a raw <preset>/ intermediate, then finalize_run.py promotes
# best_model.pth, run_info.json, metrics_history.json and train.log to the run
# root.  With SKIP_TEST=false the 20-dataset eval runs as a second stage and
# writes the canonical eval/ tree.
#
# Usage:
#   GPU=1 PRESET=full EPOCHS=300 SKIP_TEST=true \
#     bash scripts/skein/run_skein_quick.sh
#   GPU=1 PRESET=full EPOCHS=300 SKIP_TEST=false SEED=0 \
#     bash scripts/skein/run_skein_quick.sh
#   GPU=1 PRESET=sacm EPOCHS=20 MONITOR=official \
#     bash scripts/skein/run_skein_quick.sh
#
# Environment:
#   PYTHON, GPU, PRESET, RUN_NAME, EPOCHS, VAL_INTERVAL, SEED, DATASETS, CKPT,
#   TRAIN_ROOT, OUTPUT_ROOT, CANONICAL_ROOT, MONITOR, USE_WANDB, WANDB_MODE,
#   NO_AUGMENT (1 disables all train-time flips/rotations/elastic/colour;
#   default 1 = augmentation off)
#   WANDB_NAME, INIT_STAGE2_FROM_STAGE1, SKIP_TEST, SCNP, SCNP_WINDOW,
#   SCNP_ORIGINAL_WEIGHT, REUSE_EVAL_MODEL, PAPER_METRICS_ONLY, RESUME,
#   VAL_SELECTION (auto/iou/gating/quality/fixed), VAL_FIXED_HEAD,
#   VAL_EMPTY_GUARD, TOPOLOGY_LOSS, TOPOLOGY_LOSS_WEIGHT, TOPOLOGY_RESOLUTION
#
# MONITOR=baseline (default): every-epoch macro Dice/IoU/clDice/HD95 and
#   per-dataset W&B keys, exactly like src/baseline_zoo/runtime.py;
#   checkpoint selection uses macro Dice; no FOV.
# MONITOR=official: official SACM-style loss/F1/LR terminal lines, no FOV,
#   and only loss/F1 during training validation.
# MONITOR=sacm: fast validation but still prints the rich wide table (legacy).
# MONITOR=full: re-enable the full validation metric suite (Dice/IoU/clDice/
#   HD95 + AUC/Betti/Topograph engines).
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
REPO="$(pwd)"

# Reproducibility: these must be set before Python starts.
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

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
RUN_NAME="${RUN_NAME:-${PRESET}}"
source scripts/skein/wandb_env.sh
EPOCHS="${EPOCHS:-300}"
VAL_INTERVAL="${VAL_INTERVAL:-}"
SEED="${SEED:-0}"
DATASETS="${DATASETS:-}"
CKPT="${CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot_seed${SEED}}"
CANONICAL_ROOT="${CANONICAL_ROOT:-results/runs}"
FINAL_RUN_ROOT="${CANONICAL_ROOT}/skein/${PRESET}/seed_${SEED}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$FINAL_RUN_ROOT}"
# 'last'/'best' normally refer to the raw preset dir.  After finalize_run has
# promoted the checkpoint, fall back to the canonical run-root copy so a
# one-click resume still works.
if [ "${RESUME:-}" = "last" ] \
    && [ ! -f "$OUTPUT_ROOT/$PRESET/last_model.pth" ] \
    && [ -f "$FINAL_RUN_ROOT/last_model.pth" ]; then
  RESUME="$FINAL_RUN_ROOT/last_model.pth"
fi
if [ "${RESUME:-}" = "best" ] \
    && [ ! -f "$OUTPUT_ROOT/$PRESET/best_model.pth" ] \
    && [ -f "$FINAL_RUN_ROOT/best_model.pth" ]; then
  RESUME="$FINAL_RUN_ROOT/best_model.pth"
fi
if ! printf '%s\n' "$OUTPUT_ROOT" | grep -qE "seed[_-]?${SEED}([^0-9]|$)"; then
  echo "WARNING: OUTPUT_ROOT does not contain seed_${SEED} or seed${SEED}; different seeds can collide." >&2
fi
MONITOR="${MONITOR:-baseline}"
NO_AUGMENT="${NO_AUGMENT:-1}"
INIT_STAGE2_FROM_STAGE1="${INIT_STAGE2_FROM_STAGE1:-true}"
# Canonical Skein defaults reproduce the old fixed2 operating point:
# resize geometry, fixed head 2, no empty guard.  Other presets keep the
# legacy aspect_pad/IoU-selector defaults.
case "$PRESET" in
  skein|full_fixed2_aux)
    INPUT_MODE="${INPUT_MODE:-resize}"
    EVAL_INPUT_MODE="${EVAL_INPUT_MODE:-resize}"
    EMPTY_GUARD="${EMPTY_GUARD:-false}"
    SELECTION="${SELECTION:-}"
    FIXED_HEAD="${FIXED_HEAD:-2}"
    VAL_SELECTION="${VAL_SELECTION:-fixed}"
    VAL_FIXED_HEAD="${VAL_FIXED_HEAD:-2}"
    ;;
  *)
    INPUT_MODE="${INPUT_MODE:-aspect_pad}"
    EVAL_INPUT_MODE="${EVAL_INPUT_MODE:-aspect_pad}"
    EMPTY_GUARD="${EMPTY_GUARD:-true}"
    SELECTION="${SELECTION:-}"
    FIXED_HEAD="${FIXED_HEAD:-}"
    VAL_SELECTION="${VAL_SELECTION:-auto}"
    VAL_FIXED_HEAD="${VAL_FIXED_HEAD:-2}"
    ;;
esac
FG_CROP_PROB="${FG_CROP_PROB:-0.5}"
# foreground_crop_prob is only implemented by native_crop.  Resize/aspect_pad
# ignore it; record 0 so train_config does not claim a crop that never ran.
if [ "$INPUT_MODE" != "native_crop" ] && [ "$FG_CROP_PROB" != "0" ]; then
  echo "WARNING: foreground_crop_prob is ignored for input_mode=$INPUT_MODE; recording 0" >&2
  FG_CROP_PROB=0
fi
BOUNDARY_LOSS_WEIGHT="${BOUNDARY_LOSS_WEIGHT:-0.0}"
BOUNDARY_LOSS_RESOLUTION="${BOUNDARY_LOSS_RESOLUTION:-512}"
BOUNDARY_LOSS_WARMUP="${BOUNDARY_LOSS_WARMUP:-10}"
BOUNDARY_LOSS_RAMP="${BOUNDARY_LOSS_RAMP:-5}"
USE_TTA="${USE_TTA:-false}"
SELECTION_METRIC="${SELECTION_METRIC:-dice}"
VAL_EMPTY_GUARD="${VAL_EMPTY_GUARD:-$EMPTY_GUARD}"
QUALITY_SMALL_COMPONENT_WEIGHT="${QUALITY_SMALL_COMPONENT_WEIGHT:-0.5}"
QUALITY_COMPONENT_WEIGHT="${QUALITY_COMPONENT_WEIGHT:-0.05}"
TOPOLOGY_LOSS="${TOPOLOGY_LOSS:-}"
TOPOLOGY_LOSS_WEIGHT="${TOPOLOGY_LOSS_WEIGHT:-0.1}"
TOPOLOGY_RESOLUTION="${TOPOLOGY_RESOLUTION:-256}"
# Keep the one-shot eval aligned with the validation policy unless the user
# explicitly overrides it.
if [ "$VAL_SELECTION" = "fixed" ] && [ -z "$FIXED_HEAD" ]; then
  FIXED_HEAD="$VAL_FIXED_HEAD"
fi
if [ "$VAL_SELECTION" = "quality" ] && [ -z "$SELECTION" ]; then
  SELECTION="quality"
fi
COMPONENT_METRICS="${COMPONENT_METRICS:-false}"
COMPONENT_PRUNE_MODE="${COMPONENT_PRUNE_MODE:-off}"
COMPONENT_PRUNE_MIN_AREA="${COMPONENT_PRUNE_MIN_AREA:-0}"
COMPONENT_PRUNE_MIN_SKELETON="${COMPONENT_PRUNE_MIN_SKELETON:-0}"
COMPONENT_PRUNE_REL_AREA="${COMPONENT_PRUNE_REL_AREA:-0.0}"
COMPONENT_PRUNE_REL_SKELETON="${COMPONENT_PRUNE_REL_SKELETON:-0.0}"
COMPONENT_PRUNE_KEEP_LARGEST="${COMPONENT_PRUNE_KEEP_LARGEST:-1}"
AUDIT_HEADS="${AUDIT_HEADS:-false}"
AUDIT_HEADS_FULL="${AUDIT_HEADS_FULL:-false}"
SAVE_CANDIDATE_MASKS="${SAVE_CANDIDATE_MASKS:-false}"
SAVE_CANDIDATE_PROBS="${SAVE_CANDIDATE_PROBS:-false}"
AUDIT_THRESHOLDS="${AUDIT_THRESHOLDS:-}"
AUDIT_HEAD_THRESHOLDS="${AUDIT_HEAD_THRESHOLDS:-}"
SCNP="${SCNP:-false}"
SCNP_WINDOW="${SCNP_WINDOW:-3}"
SCNP_ORIGINAL_WEIGHT="${SCNP_ORIGINAL_WEIGHT:-0.0}"
DEEP_CLOSING_CKPT="${DEEP_CLOSING_CKPT:-}"
DEEP_CLOSING_THRESHOLD="${DEEP_CLOSING_THRESHOLD:-}"
SKIP_TEST="${SKIP_TEST:-false}"
RESUME="${RESUME:-}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
DETERMINISTIC="${DETERMINISTIC:-true}"
DETERMINISTIC_STRICT="${DETERMINISTIC_STRICT:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
export WANDB_MODE
if [ "$USE_WANDB" = "true" ] && [ "$WANDB_MODE" != "offline" ] && [ "$WANDB_MODE" != "disabled" ]; then
  if ! wandb_auth_ok; then
    echo "ERROR: W&B is enabled (mode=$WANDB_MODE) but no credentials were found." >&2
    echo "Set WANDB_API_KEY or ~/.netrc, or use WANDB_MODE=offline / USE_WANDB=false." >&2
    exit 1
  fi
fi
case "$MONITOR" in
  baseline) TRAIN_EXTRA=(--use_wandb "$USE_WANDB" --fast_val false
                --val_auc false --val_betti_matching false --val_topograph false
                --log_style baseline --no-fov)
              EVAL_EXTRA="--use_wandb $USE_WANDB --no-fov"
              [ -n "$VAL_INTERVAL" ] || VAL_INTERVAL=1 ;;
  official) TRAIN_EXTRA=(--use_wandb "$USE_WANDB" --fast_val true
              --val_auc false --val_betti_matching false --val_topograph false
              --log_style official --no-fov)
            EVAL_EXTRA="--use_wandb $USE_WANDB --no-fov"
            [ -n "$VAL_INTERVAL" ] || VAL_INTERVAL=10 ;;
  sacm) TRAIN_EXTRA=(--use_wandb "$USE_WANDB" --fast_val true
          --val_auc false --val_betti_matching false --val_topograph false)
        EVAL_EXTRA="--use_wandb $USE_WANDB"
        [ -n "$VAL_INTERVAL" ] || VAL_INTERVAL=10 ;;
  full) TRAIN_EXTRA=(--use_wandb "$USE_WANDB")
        EVAL_EXTRA="--use_wandb $USE_WANDB"
        [ -n "$VAL_INTERVAL" ] || VAL_INTERVAL=10 ;;
  *) echo "ERROR: MONITOR must be 'baseline', 'official', 'sacm', or 'full', got '$MONITOR'" >&2
     exit 2 ;;
esac
TRAIN_EXTRA+=(--wandb_name "${WANDB_NAME:-skein_seed${SEED}_${PRESET}}")

TRAIN_EXTRA+=(--scnp "$SCNP" --scnp_window "$SCNP_WINDOW" \
              --scnp_original_weight "$SCNP_ORIGINAL_WEIGHT")
TRAIN_EXTRA+=(--val_selection "$VAL_SELECTION" \
              --selection_metric "$SELECTION_METRIC" \
              --val_fixed_head "$VAL_FIXED_HEAD" \
              --val_empty_guard "$VAL_EMPTY_GUARD" \
              --quality_small_component_weight "$QUALITY_SMALL_COMPONENT_WEIGHT" \
              --quality_component_weight "$QUALITY_COMPONENT_WEIGHT")
if [ -n "$TOPOLOGY_LOSS" ]; then
  TRAIN_EXTRA+=(--topology_loss "$TOPOLOGY_LOSS" \
                --topology_loss_weight "$TOPOLOGY_LOSS_WEIGHT" \
                --topology_resolution "$TOPOLOGY_RESOLUTION")
fi

if [ -n "$DEEP_CLOSING_CKPT" ]; then
  if [ ! -f "$DEEP_CLOSING_CKPT" ]; then
    echo "ERROR: missing Deep Closing checkpoint: $DEEP_CLOSING_CKPT" >&2
    exit 2
  fi
  EVAL_EXTRA+=" --deep_closing_checkpoint $DEEP_CLOSING_CKPT"
  if [ -n "$DEEP_CLOSING_THRESHOLD" ]; then
    EVAL_EXTRA+=" --deep_closing_threshold $DEEP_CLOSING_THRESHOLD"
  fi
fi

TRAIN_EXTRA+=(--init_stage2_from_stage1 "$INIT_STAGE2_FROM_STAGE1")
# Official SACM trains on clean resized images (no flips/rotations/elastic/
# colour).  NO_AUGMENT=1 keeps Skein's training input distribution aligned.
if [ "$NO_AUGMENT" = "1" ]; then
  TRAIN_EXTRA+=(--no_augment)
fi

if [ ! -f "$CKPT" ]; then
  echo "ERROR: missing SAM ViT-L checkpoint: $CKPT" >&2
  exit 2
fi
if [ ! -f "$TRAIN_ROOT/protocol_manifest.csv" ]; then
  echo "ERROR: missing protocol: $TRAIN_ROOT/protocol_manifest.csv" >&2
  exit 2
fi

# Always take test roots from the same split root as training.  A seed-specific
# TRAIN_ROOT must not silently fall back to data/sacm_3shot's test sets.
SPECS="$("$PYTHON" scripts/pipeline/protocol_specs.py \
  --protocol "$TRAIN_ROOT/protocol_manifest.csv" \
  --datasets "$DATASETS")"
read -r -a spec_array <<< "$SPECS"

echo "seed=$SEED preset=$PRESET epochs=$EPOCHS gpu=$GPU monitor=$MONITOR val_interval=$VAL_INTERVAL init_stage2_from_stage1=$INIT_STAGE2_FROM_STAGE1 input_mode=$INPUT_MODE eval_input_mode=$EVAL_INPUT_MODE fg_crop_prob=$FG_CROP_PROB no_augment=$NO_AUGMENT boundary_weight=$BOUNDARY_LOSS_WEIGHT val_selection=$VAL_SELECTION selection_metric=$SELECTION_METRIC val_fixed_head=$VAL_FIXED_HEAD val_empty_guard=$VAL_EMPTY_GUARD topology=${TOPOLOGY_LOSS:-<preset>}(${TOPOLOGY_LOSS_WEIGHT}) tta=$USE_TTA empty_guard=$EMPTY_GUARD scnp=$SCNP($SCNP_WINDOW) deep_closing=${DEEP_CLOSING_CKPT:-none} skip_test=$SKIP_TEST resume=${RESUME:-none}"
echo "datasets=${#spec_array[@]}  output=$OUTPUT_ROOT"
# Training only.  Evaluation is a separate stage so every seed has one
# canonical checkpoint and one canonical eval tree.
"$PYTHON" scripts/skein/run_experiments.py \
  --gpus "$GPU" \
  --checkpoint "$CKPT" \
  --train_root "$TRAIN_ROOT" \
  --test_datasets "${spec_array[@]}" \
  --presets "$PRESET" \
  --output_root "$OUTPUT_ROOT" \
  --epochs "$EPOCHS" --val_interval "$VAL_INTERVAL" --seed "$SEED" \
  --deterministic "$DETERMINISTIC" \
  --deterministic_strict "$DETERMINISTIC_STRICT" \
  --num_workers "$NUM_WORKERS" \
  --train_input_mode "$INPUT_MODE" --eval_input_mode "$EVAL_INPUT_MODE" \
  --foreground_crop_prob "$FG_CROP_PROB" \
  --boundary_loss_weight "$BOUNDARY_LOSS_WEIGHT" \
  --boundary_loss_resolution "$BOUNDARY_LOSS_RESOLUTION" \
  --boundary_loss_warmup "$BOUNDARY_LOSS_WARMUP" \
  --boundary_loss_ramp "$BOUNDARY_LOSS_RAMP" \
  --skip_test \
  --resume "$RESUME" \
  --extra_train_args "${TRAIN_EXTRA[*]}" \
  --extra_eval_args "$EVAL_EXTRA"

echo
echo "== unified run (checkpoint promotion) =="
# This stage never evaluates; the eval stage promotes status to done.
FINALIZE_STATUS="train_only"
"$PYTHON" scripts/pipeline/finalize_run.py \
  --family skein --method skein --variant "$PRESET" --seed "$SEED" \
  --source-train "$OUTPUT_ROOT" --source-eval "$OUTPUT_ROOT/eval" \
  --out-root "$CANONICAL_ROOT" --cleanup --status "$FINALIZE_STATUS" \
  --protocol "$TRAIN_ROOT/protocol_manifest.csv" \
  || { echo "ERROR: finalize_run failed" >&2; exit 1; }

if [ ! -f "$FINAL_RUN_ROOT/best_model.pth" ] \
    && [ ! -e "$FINAL_RUN_ROOT/best_model" ]; then
  echo "ERROR: canonical checkpoint missing under $FINAL_RUN_ROOT" >&2
  exit 1
fi

echo "== artifacts =="
echo "run root:        $FINAL_RUN_ROOT"
echo "best checkpoint: $FINAL_RUN_ROOT/best_model.pth (or best_model/)"
echo "training log:    $FINAL_RUN_ROOT/train.log"
echo "run info:        $FINAL_RUN_ROOT/run_info.json"
echo "metrics history: $FINAL_RUN_ROOT/metrics_history.json"
echo "manifest:        $FINAL_RUN_ROOT/eval/run_manifest.json"

if [ "$SKIP_TEST" = "true" ]; then
  echo "== test/eval skipped (SKIP_TEST=true) =="
  exit 0
fi

echo "== evaluate 20 test sets =="
TRAIN_OUT="$FINAL_RUN_ROOT" \
OUTPUT_ROOT="$FINAL_RUN_ROOT/eval" \
CANONICAL_ROOT="$CANONICAL_ROOT" \
GPU="$GPU" SEED="$SEED" PRESET="$PRESET" TRAIN_ROOT="$TRAIN_ROOT" \
DATASETS="$DATASETS" SAM_CKPT="$CKPT" \
USE_WANDB="$USE_WANDB" WANDB_MODE="$WANDB_MODE" \
EVAL_INPUT_MODE="$EVAL_INPUT_MODE" USE_TTA="$USE_TTA" \
EMPTY_GUARD="$EMPTY_GUARD" SELECTION="$SELECTION" FIXED_HEAD="$FIXED_HEAD" \
QUALITY_SMALL_COMPONENT_WEIGHT="$QUALITY_SMALL_COMPONENT_WEIGHT" \
QUALITY_COMPONENT_WEIGHT="$QUALITY_COMPONENT_WEIGHT" \
COMPONENT_METRICS="$COMPONENT_METRICS" \
COMPONENT_PRUNE_MODE="$COMPONENT_PRUNE_MODE" \
COMPONENT_PRUNE_MIN_AREA="$COMPONENT_PRUNE_MIN_AREA" \
COMPONENT_PRUNE_MIN_SKELETON="$COMPONENT_PRUNE_MIN_SKELETON" \
COMPONENT_PRUNE_REL_AREA="$COMPONENT_PRUNE_REL_AREA" \
COMPONENT_PRUNE_REL_SKELETON="$COMPONENT_PRUNE_REL_SKELETON" \
COMPONENT_PRUNE_KEEP_LARGEST="$COMPONENT_PRUNE_KEEP_LARGEST" \
AUDIT_HEADS="$AUDIT_HEADS" AUDIT_HEADS_FULL="$AUDIT_HEADS_FULL" \
SAVE_CANDIDATE_MASKS="$SAVE_CANDIDATE_MASKS" \
SAVE_CANDIDATE_PROBS="$SAVE_CANDIDATE_PROBS" \
AUDIT_THRESHOLDS="$AUDIT_THRESHOLDS" \
AUDIT_HEAD_THRESHOLDS="$AUDIT_HEAD_THRESHOLDS" \
DEEP_CLOSING_CKPT="$DEEP_CLOSING_CKPT" \
DEEP_CLOSING_THRESHOLD="$DEEP_CLOSING_THRESHOLD" \
DETERMINISTIC="$DETERMINISTIC" DETERMINISTIC_STRICT="$DETERMINISTIC_STRICT" \
NUM_WORKERS="$NUM_WORKERS" \
REUSE_EVAL_MODEL="${REUSE_EVAL_MODEL:-0}" \
PAPER_METRICS_ONLY="${PAPER_METRICS_ONLY:-0}" \
bash scripts/skein/eval_skein_20.sh
