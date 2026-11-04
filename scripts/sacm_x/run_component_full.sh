#!/usr/bin/env bash
# One-click full pipeline (one training seed per GPU):
#   from scratch, gate-only SACM-X, then support-calibrated component-repair
#   evaluation on 20 datasets, plus a no-gate component-repair ablation.
#
# Usage:
#   bash scripts/sacm_x/run_component_full.sh
#   EPOCHS=800 TRAIN_SEEDS="0 1 2 3 4 5 6 7" \
#     bash scripts/sacm_x/run_component_full.sh
#   PER_GPU_DATA_SEED=1 bash scripts/sacm_x/run_component_full.sh
#   RUN_NO_GATE=0 bash scripts/sacm_x/run_component_full.sh
#
# Defaults:
#   PRESET=full_fixed2_gate (full_fixed2 + gate-only supervision)
#   TRAIN_SEEDS="0 1 2 3 4" (one seed per GPU; up to 8 supported)
#   RUN_NO_GATE=0 (final-model run first; enable later for ablations)
#   DATA_SEED=0 (shared data split; training seeds differ)
#   NO_AUGMENT=1 (match SACM official: no train-time geometric/colour aug)
#   EPOCHS=800, VAL_INTERVAL=1, SKIP_TEST=true during training
#
# Output:
#   results/runs/component_full_<timestamp>/
#     train/                         canonical SACM-X checkpoints
#     component_eval/seed_<s>/       gate + component prune
#     component_eval_no_gate/seed_<s>/  no-gate feature ablation
#     summary_seeds.csv             mean/std over seeds
set -euo pipefail

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
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PRESET="${PRESET:-full_fixed2_gate}"
DATA_SEED="${DATA_SEED:-0}"
PER_GPU_DATA_SEED="${PER_GPU_DATA_SEED:-0}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-800}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
USE_WANDB="${USE_WANDB:-true}"
NO_AUGMENT="${NO_AUGMENT:-1}"
SELECTION_METRIC="${SELECTION_METRIC:-dice}"
RUN_NO_GATE="${RUN_NO_GATE:-0}"
LIMIT="${LIMIT:-0}"
SAM_CKPT="${SAM_CKPT:-checkpoints/sam_vit_l_0b3195.pth}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-results/runs/component_full_${TIMESTAMP}}"
TRAIN_OUT="$OUT_ROOT/train"
EVAL_OUT="$OUT_ROOT/component_eval"
EVAL_NO_GATE_OUT="$OUT_ROOT/component_eval_no_gate"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$TRAIN_OUT" "$EVAL_OUT" "$LOG_DIR"

if [ ! -f "$SAM_CKPT" ]; then
  echo "ERROR: missing SAM checkpoint $SAM_CKPT" >&2
  exit 2
fi

echo "============================================================"
echo " full component pipeline, one seed per GPU"
echo " preset:       $PRESET"
echo " out:          $OUT_ROOT"
echo " data seed:    $DATA_SEED (per_gpu=$PER_GPU_DATA_SEED)"
echo " train seeds:  $TRAIN_SEEDS"
echo " epochs:       $EPOCHS"
echo " wandb:        $USE_WANDB"
echo " augmentation: $([ "$NO_AUGMENT" = "1" ] && echo disabled || echo enabled)"
echo " selection:    fixed head 2 / $SELECTION_METRIC"
echo " no-gate eval: $RUN_NO_GATE"
echo " python:       $PY"
echo "============================================================"

declare -a TRAIN_PIDS=()
declare -a TRAIN_NAMES=()
idx=0
for seed in $TRAIN_SEEDS; do
  if [ "$idx" -ge 8 ]; then
    echo "ERROR: more than 8 training seeds; this pipeline supports at most 8" >&2
    exit 2
  fi
  gpu="$idx"
  if [ "$PER_GPU_DATA_SEED" = "1" ]; then
    data_root="data/sacm_3shot_seed${seed}"
  else
    data_root="data/sacm_3shot_seed${DATA_SEED}"
  fi
  if [ ! -f "$data_root/protocol_manifest.csv" ]; then
    echo "ERROR: missing protocol $data_root/protocol_manifest.csv" >&2
    exit 2
  fi
  log="$LOG_DIR/train_seed${seed}.log"
  echo "[gpu $gpu] train $PRESET seed $seed data=$data_root"
  (
    # run_experiments.py forwards this id by setting CUDA_VISIBLE_DEVICES
    # for the trainer child, so it must be the physical GPU index.  Using
    # GPU=0 here made every seed override its parent mask and train on GPU0.
    export GPU="$gpu"
    export SEED="$seed"
    export PRESET="$PRESET"
    export EPOCHS="$EPOCHS"
    export VAL_INTERVAL="$VAL_INTERVAL"
    export TRAIN_ROOT="$data_root"
    export CANONICAL_ROOT="$TRAIN_OUT"
    export RESUME=""
    export SKIP_TEST=true
    export USE_WANDB="$USE_WANDB"
    export NO_AUGMENT="$NO_AUGMENT"
    export MONITOR=baseline
    export INPUT_MODE=resize
    export EVAL_INPUT_MODE=resize
    export FG_CROP_PROB=0
    export VAL_SELECTION=fixed
    export VAL_FIXED_HEAD=2
    export FIXED_HEAD=2
    export SELECTION_METRIC="$SELECTION_METRIC"
    export WANDB_NAME="component_${PRESET}_seed${seed}_${TIMESTAMP}"
    export CUDA_VISIBLE_DEVICES="$gpu"
    bash scripts/sacm_x/run_pepa_variant.sh "$PRESET" train
  ) > "$log" 2>&1 &
  TRAIN_PIDS+=("$!")
  TRAIN_NAMES+=("seed${seed}")
  idx=$((idx + 1))
done

train_fail=0
for i in "${!TRAIN_PIDS[@]}"; do
  if wait "${TRAIN_PIDS[$i]}"; then
    echo "TRAIN DONE ${TRAIN_NAMES[$i]}"
  else
    echo "TRAIN FAILED ${TRAIN_NAMES[$i]} (log: $LOG_DIR/train_${TRAIN_NAMES[$i]}.log)" >&2
    train_fail=1
  fi
done
if [ "$train_fail" != "0" ]; then
  echo "Stopping: training failures." >&2
  exit 3
fi

resolve_ckpt() {
  local seed="$1" cand
  for cand in \
    "$TRAIN_OUT/sacm_x/$PRESET/seed_$seed/best_model.pth" \
    "$TRAIN_OUT/sacm_x/$PRESET/seed_$seed/best_model" \
    "$TRAIN_OUT/sacm_x/$PRESET/seed_${seed}/best_model.pth"; do
    [ -e "$cand" ] && { printf '%s\n' "$cand"; return 0; }
  done
  return 1
}

# Evaluate each seed on its own GPU; gate and no-gate variants sequentially.
declare -a EVAL_PIDS=()
declare -a EVAL_NAMES=()
idx=0
for seed in $TRAIN_SEEDS; do
  gpu="$idx"
  ckpt="$(resolve_ckpt "$seed")" || {
    echo "ERROR: missing canonical checkpoint for seed $seed under $TRAIN_OUT" >&2
    exit 4
  }
  if [ "$PER_GPU_DATA_SEED" = "1" ]; then
    data_root="data/sacm_3shot_seed${seed}"
  else
    data_root="data/sacm_3shot_seed${DATA_SEED}"
  fi
  gate_out="$EVAL_OUT/seed_${seed}"
  no_gate_out="$EVAL_NO_GATE_OUT/seed_${seed}"
  log="$LOG_DIR/eval_seed${seed}.log"
  echo "[gpu $gpu] eval seed $seed ($ckpt)"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" scripts/sacm_x/component_repair_eval.py \
      --preset "$PRESET" \
      --checkpoint "$ckpt" \
      --data-root "$data_root" \
      --test-root "$data_root/test" \
      --output-dir "$gate_out" \
      --device cuda:0 \
      --limit "$LIMIT" \
      --run-name "seed${seed}_gate"
    if [ "$RUN_NO_GATE" = "1" ]; then
      "$PY" scripts/sacm_x/component_repair_eval.py \
        --preset "$PRESET" \
        --checkpoint "$ckpt" \
        --data-root "$data_root" \
        --test-root "$data_root/test" \
        --output-dir "$no_gate_out" \
        --device cuda:0 \
        --limit "$LIMIT" \
        --drop-gate-features \
        --run-name "seed${seed}_no_gate"
    fi
  ) > "$log" 2>&1 &
  EVAL_PIDS+=("$!")
  EVAL_NAMES+=("seed${seed}")
  idx=$((idx + 1))
done

eval_fail=0
for i in "${!EVAL_PIDS[@]}"; do
  if wait "${EVAL_PIDS[$i]}"; then
    echo "EVAL DONE ${EVAL_NAMES[$i]}"
  else
    echo "EVAL FAILED ${EVAL_NAMES[$i]} (log: $LOG_DIR/eval_${EVAL_NAMES[$i]}.log)" >&2
    eval_fail=1
  fi
done
if [ "$eval_fail" != "0" ]; then
  echo "Stopping: evaluation failures." >&2
  exit 5
fi

"$PY" - "$OUT_ROOT" <<'PY'
import csv, json, math, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for variant, base in (('gate', root / 'component_eval'),
                      ('no_gate', root / 'component_eval_no_gate')):
    if not base.exists():
        continue
    for p in sorted(base.glob('seed_*/summary.json')):
        d = json.loads(p.read_text())
        seed = p.parent.name.replace('seed_', '')
        for role, m in d.items():
            rows.append({'variant': variant, 'seed': seed, 'role': role,
                         **{k: m.get(k, float('nan')) for k in (
                             'n_datasets', 'coarse_dice', 'prune_dice',
                             'coarse_cldice', 'prune_cldice',
                             'coarse_hd95', 'prune_hd95',
                             'coarse_components', 'prune_components',
                             'coarse_cc_error', 'prune_cc_error',
                             'gate_auc')}})
out = root / 'summary_seeds.csv'
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
    if rows:
        w.writeheader(); w.writerows(rows)
print(f'wrote {out} ({len(rows)} rows)')
for variant in ('gate', 'no_gate'):
    print('\n===', variant, '===')
    for role in ('Seen', 'Unseen', 'Novel', 'All'):
        sub = [r for r in rows if r['variant'] == variant and r['role'] == role]
        if not sub:
            continue
        def ms(key):
            vals = [r[key] for r in sub if math.isfinite(r[key])]
            return (statistics.fmean(vals),
                    statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        d, ds = ms('prune_dice'); cd, cds = ms('coarse_dice')
        c, cs = ms('prune_cldice'); cc, ccs = ms('coarse_cldice')
        print(f'{role:6s} n={len(sub):2d} '
              f'Dice {cd:.4f}+-{cds:.4f} -> {d:.4f}+-{ds:.4f} '
              f'clDice {cc:.4f} -> {c:.4f} '
              f'ccerr {ms("coarse_cc_error")[0]:.1f} -> {ms("prune_cc_error")[0]:.1f}')
PY

echo "Full pipeline done: $OUT_ROOT"
