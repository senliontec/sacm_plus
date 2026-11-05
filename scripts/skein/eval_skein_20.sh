#!/usr/bin/env bash
# Evaluate the saved best Skein checkpoint on the 20 frozen-protocol test sets.
# No retraining.
#
# Usage:
#   SEED=0 PRESET=full_typed bash scripts/skein/eval_skein_20.sh
#   GPU=2 SEED=0 PRESET=full_typed bash scripts/skein/eval_skein_20.sh
#   DATASETS="DRIVE WIRE" bash scripts/skein/eval_skein_20.sh
#
# Environment:
#   GPU (default 3), TRAIN_OUT, CKPT, OUTPUT_ROOT, CANONICAL_ROOT,
#   PRESET (default full), SEED, SAM_CKPT, DATASETS, USE_WANDB, WANDB_MODE,
#   SELECTION (auto/iou/gating/quality), FIXED_HEAD (0-3),
#   QUALITY_SMALL_COMPONENT_WEIGHT, QUALITY_COMPONENT_WEIGHT,
#   SAVE_CANDIDATE_MASKS, SAVE_CANDIDATE_PROBS, REUSE_EVAL_MODEL,
#   PAPER_METRICS_ONLY.
#
# Canonical layout:
#   TRAIN_OUT=results/runs/skein/<preset>/seed<seed>
#   OUTPUT_ROOT=$TRAIN_OUT/eval
#   CKPT=$TRAIN_OUT/best_model.pth
# The raw <eval>/<preset>/<DATASET> tree is flattened into
# <eval>/<DATASET> and the training checkpoint is promoted by
# finalize_run.py; no 1.4 GiB copy is made.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

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

GPU="${GPU:-3}"
PRESET="${PRESET:-full}"
SEED="${SEED:-0}"
CANONICAL_ROOT="${CANONICAL_ROOT:-results/runs}"
TRAIN_OUT="${TRAIN_OUT:-$CANONICAL_ROOT/skein/$PRESET/seed_$SEED}"
# Backward compatibility: <run>/train was the old training subdirectory.
if [[ "$TRAIN_OUT" == */train ]]; then
  TRAIN_OUT="${TRAIN_OUT%/train}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$TRAIN_OUT/eval}"
CKPT="${CKPT:-$TRAIN_OUT/best_model.pth}"
if [ ! -f "$CKPT" ] && [ -f "$TRAIN_OUT/$PRESET/best_model.pth" ]; then
  CKPT="$TRAIN_OUT/$PRESET/best_model.pth"
fi
if [ -n "$TRAIN_OUT" ] && [ "$OUTPUT_ROOT" = "$TRAIN_OUT" ]; then
  echo "ERROR: OUTPUT_ROOT must differ from TRAIN_OUT" >&2
  echo "Use TRAIN_OUT=<run-root> and OUTPUT_ROOT=<run-root>/eval." >&2
  exit 2
fi
if ! printf '%s\n' "$OUTPUT_ROOT" | grep -qE "seed[_-]?${SEED}([^0-9]|$)"; then
  echo "WARNING: OUTPUT_ROOT does not contain seed_${SEED} or seed${SEED}; different seeds can collide." >&2
fi
ckpt_seed_num=$(printf '%s\n' "$CKPT" \
  | grep -oE 'seed[_-]?[0-9]+' | tail -n 1 | sed -E 's/seed[_-]?//' || true)
if [ -n "$ckpt_seed_num" ] && [ "$ckpt_seed_num" != "$SEED" ]; then
  echo "ERROR: CKPT seed (${ckpt_seed_num}) != SEED (${SEED})" >&2
  echo "Use SEED=${ckpt_seed_num} or a checkpoint from seed_${SEED}." >&2
  exit 2
fi
SAM_CKPT="${SAM_CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
DATASETS="${DATASETS:-}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
DETERMINISTIC="${DETERMINISTIC:-true}"
DETERMINISTIC_STRICT="${DETERMINISTIC_STRICT:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# W&B must not be initialized in the parent when DataLoader forks: the
# worker inherits the W&B socket and raises ForkedError, failing every
# dataset.  Evaluation does not need W&B; disable it when workers > 0.
if { [ "$USE_WANDB" = "true" ] || [ "$USE_WANDB" = "1" ]; } \
    && [ "${NUM_WORKERS:-0}" -gt 0 ]; then
  echo "WARNING: disabling W&B for evaluation (DataLoader workers > 0);" \
       "set NUM_WORKERS=0 to keep W&B." >&2
  USE_WANDB=false
fi
TRAIN_ROOT="${TRAIN_ROOT:-data/sacm_3shot_seed${SEED}}"
EVAL_INPUT_MODE="${EVAL_INPUT_MODE:-aspect_pad}"
USE_TTA="${USE_TTA:-false}"
EMPTY_GUARD="${EMPTY_GUARD:-true}"
# Optional speed-ups; both default to the previous behavior.
#   REUSE_EVAL_MODEL=1  load the model/checkpoint once for all datasets
#   PAPER_METRICS_ONLY=1 compute only Dice/IoU/clDice/HD95
REUSE_EVAL_MODEL="${REUSE_EVAL_MODEL:-0}"
PAPER_METRICS_ONLY="${PAPER_METRICS_ONLY:-0}"
# Fairness ablation: baselines do not have Skein's aspect_pad geometry or
# empty-mask fallback.  Set MATCH_BASELINE_GEOMETRY=1 to evaluate Skein
# under the same square-resize / no-empty-guard operating point.
MATCH_BASELINE_GEOMETRY="${MATCH_BASELINE_GEOMETRY:-0}"
if [ "$MATCH_BASELINE_GEOMETRY" = "1" ]; then
  EVAL_INPUT_MODE=resize
  EMPTY_GUARD=false
fi
SELECTION="${SELECTION:-}"
FIXED_HEAD="${FIXED_HEAD:-}"
QUALITY_SMALL_COMPONENT_WEIGHT="${QUALITY_SMALL_COMPONENT_WEIGHT:-0.5}"
QUALITY_COMPONENT_WEIGHT="${QUALITY_COMPONENT_WEIGHT:-0.05}"
AUDIT_HEADS="${AUDIT_HEADS:-false}"
AUDIT_HEADS_FULL="${AUDIT_HEADS_FULL:-false}"
SAVE_CANDIDATE_MASKS="${SAVE_CANDIDATE_MASKS:-false}"
SAVE_CANDIDATE_PROBS="${SAVE_CANDIDATE_PROBS:-false}"
COMPONENT_METRICS="${COMPONENT_METRICS:-false}"
COMPONENT_PRUNE_MODE="${COMPONENT_PRUNE_MODE:-off}"
COMPONENT_PRUNE_MIN_AREA="${COMPONENT_PRUNE_MIN_AREA:-0}"
COMPONENT_PRUNE_MIN_SKELETON="${COMPONENT_PRUNE_MIN_SKELETON:-0}"
COMPONENT_PRUNE_REL_AREA="${COMPONENT_PRUNE_REL_AREA:-0.0}"
COMPONENT_PRUNE_REL_SKELETON="${COMPONENT_PRUNE_REL_SKELETON:-0.0}"
COMPONENT_PRUNE_KEEP_LARGEST="${COMPONENT_PRUNE_KEEP_LARGEST:-1}"
AUDIT_THRESHOLDS="${AUDIT_THRESHOLDS:-}"
AUDIT_HEAD_THRESHOLDS="${AUDIT_HEAD_THRESHOLDS:-}"
DEEP_CLOSING_CKPT="${DEEP_CLOSING_CKPT:-}"
DEEP_CLOSING_THRESHOLD="${DEEP_CLOSING_THRESHOLD:-}"
SKIP_EVAL="${SKIP_EVAL:-false}"
export WANDB_MODE
source scripts/skein/wandb_env.sh

if [ ! -f "$CKPT" ]; then
  echo "ERROR: missing checkpoint: $CKPT" >&2
  echo "Train first or set OUTPUT_ROOT/CKPT correctly." >&2
  exit 2
fi
if [ ! -f "$TRAIN_ROOT/protocol_manifest.csv" ]; then
  echo "ERROR: missing protocol: $TRAIN_ROOT/protocol_manifest.csv" >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || used="?"
  echo "GPU $GPU memory used: ${used:-?} MiB"
  if [[ "${used:-0}" =~ ^[0-9]+$ ]] && [ "$used" -gt 2000 ]; then
    echo "ERROR: GPU $GPU already has ${used} MiB in use; refusing to start evaluation." >&2
    echo "Free it or run on another card, e.g. GPU=2 bash scripts/skein/eval_skein_20.sh" >&2
    nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null || true
    exit 3
  fi
fi

# Always take test roots from the same split root as training.  A seed-specific
# TRAIN_ROOT must not silently fall back to data/sacm_3shot's test sets.
SPECS="$("$PYTHON" scripts/pipeline/protocol_specs.py \
  --protocol "$TRAIN_ROOT/protocol_manifest.csv" \
  --datasets "$DATASETS")"
read -r -a spec_array <<< "$SPECS"

echo "seed=$SEED"
echo "train_out=${TRAIN_OUT:-<none>}"
echo "checkpoint=$CKPT"
echo "eval_root=$OUTPUT_ROOT"
echo "datasets=${#spec_array[@]}"
if [ -n "$DEEP_CLOSING_CKPT" ]; then
  if [ ! -f "$DEEP_CLOSING_CKPT" ]; then
    echo "ERROR: missing Deep Closing checkpoint: $DEEP_CLOSING_CKPT" >&2
    exit 2
  fi
  echo "deep_closing=$DEEP_CLOSING_CKPT"
fi

if [ "$SKIP_EVAL" = "true" ] || [ "$SKIP_EVAL" = "1" ]; then
  echo "SKIP_EVAL=$SKIP_EVAL: using existing per-dataset results"
  rc=0
else
  echo "no retraining; evaluating with --skip_train"
  echo "eval_input_mode=$EVAL_INPUT_MODE tta=$USE_TTA empty_guard=$EMPTY_GUARD"
  echo "reuse_eval_model=$REUSE_EVAL_MODEL paper_metrics_only=$PAPER_METRICS_ONLY"
  EXTRA_EVAL="--use_wandb $USE_WANDB --no-fov"
  if [ -n "$DEEP_CLOSING_CKPT" ]; then
    EXTRA_EVAL+=" --deep_closing_checkpoint $DEEP_CLOSING_CKPT"
    if [ -n "$DEEP_CLOSING_THRESHOLD" ]; then
      EXTRA_EVAL+=" --deep_closing_threshold $DEEP_CLOSING_THRESHOLD"
    fi
  fi
  EXTRA_EVAL+=" --quality_small_component_weight $QUALITY_SMALL_COMPONENT_WEIGHT"
  EXTRA_EVAL+=" --quality_component_weight $QUALITY_COMPONENT_WEIGHT"
  if [ "$COMPONENT_METRICS" = "true" ] || [ "$COMPONENT_METRICS" = "1" ]; then
    EXTRA_EVAL+=" --component_metrics"
  fi
  if [ "$COMPONENT_PRUNE_MODE" != "off" ]; then
    EXTRA_EVAL+=" --component_prune_mode $COMPONENT_PRUNE_MODE"
    EXTRA_EVAL+=" --component_prune_min_area $COMPONENT_PRUNE_MIN_AREA"
    EXTRA_EVAL+=" --component_prune_min_skeleton $COMPONENT_PRUNE_MIN_SKELETON"
    EXTRA_EVAL+=" --component_prune_rel_area $COMPONENT_PRUNE_REL_AREA"
    EXTRA_EVAL+=" --component_prune_rel_skeleton $COMPONENT_PRUNE_REL_SKELETON"
    EXTRA_EVAL+=" --component_prune_keep_largest $COMPONENT_PRUNE_KEEP_LARGEST"
  fi
  if [ "$AUDIT_HEADS" = "true" ] || [ "$AUDIT_HEADS" = "1" ]; then
    EXTRA_EVAL+=" --audit_heads"
  fi
  if [ "$AUDIT_HEADS_FULL" = "true" ] || [ "$AUDIT_HEADS_FULL" = "1" ]; then
    EXTRA_EVAL+=" --audit_heads_full"
  fi
  if [ "$SAVE_CANDIDATE_MASKS" = "true" ] || [ "$SAVE_CANDIDATE_MASKS" = "1" ]; then
    EXTRA_EVAL+=" --save_candidate_masks"
  fi
  if [ "$SAVE_CANDIDATE_PROBS" = "true" ] || [ "$SAVE_CANDIDATE_PROBS" = "1" ]; then
    EXTRA_EVAL+=" --save_candidate_probs"
  fi
  if [ -n "$AUDIT_THRESHOLDS" ]; then
    EXTRA_EVAL+=" --audit_thresholds $AUDIT_THRESHOLDS"
  fi
  if [ -n "$AUDIT_HEAD_THRESHOLDS" ]; then
    EXTRA_EVAL+=" --audit_head_thresholds $AUDIT_HEAD_THRESHOLDS"
  fi
  EXTRA_RUN=(--eval_input_mode "$EVAL_INPUT_MODE"
    --deterministic "$DETERMINISTIC"
    --deterministic_strict "$DETERMINISTIC_STRICT"
    --num_workers "$NUM_WORKERS")
  if [ -n "$SELECTION" ]; then
    EXTRA_RUN+=(--selection "$SELECTION")
  fi
  if [ -n "$FIXED_HEAD" ]; then
    EXTRA_RUN+=(--fixed_head "$FIXED_HEAD")
  fi
  if [ "$EMPTY_GUARD" = "true" ] || [ "$EMPTY_GUARD" = "1" ]; then
    EXTRA_RUN+=(--empty_guard)
  fi
  if [ "$USE_TTA" = "true" ] || [ "$USE_TTA" = "1" ]; then
    EXTRA_RUN+=(--tta)
  fi
  if [ "$REUSE_EVAL_MODEL" = "true" ] || [ "$REUSE_EVAL_MODEL" = "1" ]; then
    EXTRA_RUN+=(--reuse_eval_model)
  fi
  if [ "$PAPER_METRICS_ONLY" = "true" ] || [ "$PAPER_METRICS_ONLY" = "1" ]; then
    EXTRA_RUN+=(--paper_metrics_only)
  fi
  "$PYTHON" scripts/skein/run_experiments.py \
    --gpus "$GPU" \
    --checkpoint "$SAM_CKPT" \
    --train_root "$TRAIN_ROOT" \
    --test_datasets "${spec_array[@]}" \
    --presets "$PRESET" \
    --output_root "$OUTPUT_ROOT" \
    --seed "$SEED" \
    --skip_train \
    --trained_weights "$CKPT" \
    "${EXTRA_RUN[@]}" \
    --extra_eval_args "$EXTRA_EVAL"
  rc=$?
fi

if [ "$rc" -ne 0 ]; then
  echo "ERROR: one or more dataset evaluations failed." >&2
  echo "Refusing to overwrite the existing canonical eval tables with a partial set." >&2
  echo "Raw partial outputs (if any): $OUTPUT_ROOT/$PRESET/" >&2
  exit 1
fi

echo
echo "== aggregate =="
agg_rc=0
if [ "$SKIP_EVAL" = "true" ] || [ "$SKIP_EVAL" = "1" ]; then
  if [ -f "$OUTPUT_ROOT/tables/all_metrics.csv" ] \
      && [ ! -d "$OUTPUT_ROOT/$PRESET" ]; then
    echo "using existing canonical tables (raw $PRESET/ already finalized)"
  else
    AGG_ARGS=(--root "$OUTPUT_ROOT" --out "$OUTPUT_ROOT/tables"
      --preset "$PRESET")
    if [ -f "$TRAIN_ROOT/protocol_summary.json" ]; then
      AGG_ARGS+=(--protocol-summary "$TRAIN_ROOT/protocol_summary.json")
    fi
    if [ -n "$DATASETS" ]; then
      # shellcheck disable=SC2206
      AGG_ARGS+=(--datasets $DATASETS)
    fi
    "$PYTHON" scripts/skein/aggregate_results.py \
      "${AGG_ARGS[@]}" || agg_rc=$?
  fi
else
  AGG_ARGS=(--root "$OUTPUT_ROOT" --out "$OUTPUT_ROOT/tables"
    --preset "$PRESET")
  if [ -f "$TRAIN_ROOT/protocol_summary.json" ]; then
    AGG_ARGS+=(--protocol-summary "$TRAIN_ROOT/protocol_summary.json")
  fi
  if [ -n "$DATASETS" ]; then
    # shellcheck disable=SC2206
    AGG_ARGS+=(--datasets $DATASETS)
  fi
  "$PYTHON" scripts/skein/aggregate_results.py \
    "${AGG_ARGS[@]}" || agg_rc=$?
fi

echo
echo "== final report =="
report_rc=0
"$PYTHON" scripts/skein/make_final_report.py \
  --tables-dir "$OUTPUT_ROOT/tables" \
  --preset "$PRESET" \
  --checkpoint "$CKPT" || report_rc=$?

if [ "$rc" -ne 0 ] || [ "$agg_rc" -ne 0 ] || [ "$report_rc" -ne 0 ]; then
  echo "ERROR: eval/aggregate/report failed (eval=$rc aggregate=$agg_rc report=$report_rc)" >&2
  exit 1
fi

echo
echo "== unified run =="
FINAL_RUN_ROOT="$CANONICAL_ROOT/skein/$PRESET/seed_$SEED"
"$PYTHON" scripts/pipeline/finalize_run.py \
  --family skein --method skein --variant "$PRESET" --seed "$SEED" \
  --source-train "$TRAIN_OUT" --source-eval "$OUTPUT_ROOT" \
  --out-root "$CANONICAL_ROOT" --cleanup \
  --checkpoint "$CKPT" \
  --protocol "$TRAIN_ROOT/protocol_manifest.csv" \
  || { echo "ERROR: finalize_run failed" >&2; exit 1; }

echo
echo "== results =="
echo "run root:    $FINAL_RUN_ROOT"
echo "checkpoint:  $FINAL_RUN_ROOT/best_model.pth (or best_model/)"
echo "eval log:    $FINAL_RUN_ROOT/eval/eval.log"
echo "jobs:        $FINAL_RUN_ROOT/eval/jobs.csv"
echo "manifest:    $FINAL_RUN_ROOT/eval/run_manifest.json"
echo "per-dataset: $FINAL_RUN_ROOT/eval/<DATASET>/metrics.csv"
echo "csv:         $FINAL_RUN_ROOT/eval/tables/all_metrics.csv"
echo "macro:       $FINAL_RUN_ROOT/eval/tables/macro_metrics.csv"
echo "report:      $FINAL_RUN_ROOT/eval/tables/final_report.md"
exit 0
