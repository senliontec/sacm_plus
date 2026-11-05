#!/usr/bin/env bash
# Re-evaluate EXISTING final checkpoints on all 20 protocol test datasets.
#
# This does not train.  It fixes the old component evaluator's blind spot:
# data/sacm_3shot_seedN/test contains only the six Seen test dirs; Unseen and
# Novel roots are resolved from protocol_manifest.csv.
#
# Usage (on the server, from the repo root):
#   bash scripts/sacm_x/reeval_final_component_all20.sh
#
# Defaults:
#   FINAL_ROOT = latest results/runs/final_gpu0_4_* directory
#   PRESET=full_fixed2_gate
#   SEEDS="0 1 2 3 4"
#   GPU_LIST="0 1 2 3 4"
#   OUT_ROOT=$FINAL_ROOT/component_eval_all20_<timestamp>
#
# Output:
#   <OUT_ROOT>/seed_<s>/per_dataset.csv
#   <OUT_ROOT>/seed_<s>/summary.json
#   <OUT_ROOT>/summary_seeds.csv
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

# Do not inherit a stale PRESET=full from an interactive shell.  The final
# checkpoint lives under full_fixed2_gate; use REEVAL_PRESET for an explicit
# override.
PRESET="${REEVAL_PRESET:-full_fixed2_gate}"
SEEDS="${SEEDS:-0 1 2 3 4}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4}"
LIMIT="${LIMIT:-0}"
DATASETS="${DATASETS:-}"
FINAL_ROOT="${FINAL_ROOT:-}"
if [ -z "$FINAL_ROOT" ]; then
  FINAL_ROOT="$(ls -dt results/runs/final_gpu0_4_* 2>/dev/null | head -1 || true)"
fi
if [ -z "$FINAL_ROOT" ] || [ ! -d "$FINAL_ROOT" ]; then
  echo "ERROR: final run directory not found; set FINAL_ROOT=..." >&2
  exit 2
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$FINAL_ROOT/component_eval_all20_${STAMP}}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

read -r -a SEED_ARGS <<< "$SEEDS"
read -r -a GPU_ARGS <<< "$GPU_LIST"
if [ "${#GPU_ARGS[@]}" -lt "${#SEED_ARGS[@]}" ]; then
  echo "ERROR: need ${#SEED_ARGS[@]} GPUs, got ${#GPU_ARGS[@]}" >&2
  exit 2
fi

resolve_ckpt() {
  local seed="$1" cand
  for cand in \
    "$FINAL_ROOT/train/sacm_x/$PRESET/seed_$seed/best_model.pth" \
    "$FINAL_ROOT/train/sacm_x/$PRESET/seed_${seed}/best_model.pth"; do
    [ -f "$cand" ] && { printf '%s\n' "$cand"; return 0; }
  done
  return 1
}

echo "============================================================"
echo " Re-evaluate final component repair on 20 datasets"
echo " final:  $FINAL_ROOT"
echo " preset: $PRESET"
echo " seeds:  $SEEDS"
echo " gpus:   $GPU_LIST"
echo " out:    $OUT_ROOT"
echo " limit:  $LIMIT (0 = all images)"
echo "============================================================"

PIDS=()
NAMES=()
idx=0
for seed in "${SEED_ARGS[@]}"; do
  gpu="${GPU_ARGS[$idx]}"
  ckpt="$(resolve_ckpt "$seed")" || {
    echo "ERROR: missing checkpoint for seed $seed under $FINAL_ROOT" >&2
    exit 3
  }
  data_root="data/sacm_3shot_seed${seed}"
  protocol="$data_root/protocol_manifest.csv"
  if [ ! -f "$protocol" ]; then
    echo "ERROR: missing protocol $protocol" >&2
    exit 3
  fi
  log="$LOG_DIR/eval_seed${seed}.log"
  echo "[gpu $gpu] seed $seed: $ckpt"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    extra=()
    if [ -n "$DATASETS" ]; then
      extra+=(--datasets "$DATASETS")
    fi
    "$PY" scripts/sacm_x/component_repair_eval.py \
      --preset "$PRESET" \
      --checkpoint "$ckpt" \
      --data-root "$data_root" \
      --protocol "$protocol" \
      --expected-datasets 20 \
      --output-dir "$OUT_ROOT/seed_${seed}" \
      --device cuda:0 \
      --limit "$LIMIT" \
      "${extra[@]}" \
      --run-name "seed${seed}_gate"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("seed${seed}")
  idx=$((idx + 1))
done

fail=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "EVAL DONE ${NAMES[$i]}"
  else
    echo "EVAL FAILED ${NAMES[$i]} (log: $LOG_DIR/eval_${NAMES[$i]}.log)" >&2
    fail=1
  fi
done
if [ "$fail" != "0" ]; then
  echo "Stopping: evaluation failures." >&2
  exit 4
fi

"$PY" - "$OUT_ROOT" <<'PY'
import csv, json, math, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
keys = (
    'n_datasets', 'coarse_dice', 'prune_dice',
    'coarse_iou', 'prune_iou',
    'coarse_cldice', 'prune_cldice',
    'coarse_hd95', 'prune_hd95',
    'coarse_components', 'prune_components',
    'coarse_cc_error', 'prune_cc_error',
    'gate_auc', 'gate_auprc',
    'gate_f1_0_5', 'gate_precision_0_5', 'gate_recall_0_5',
    'gate_iou_0_5', 'gate_best_f1', 'gate_best_iou',
    'gate_best_threshold', 'gate_auc_topo', 'gate_auprc_topo',
    'gate_f1_0_5_topo',
)
rows = []
for p in sorted(root.glob('seed_*/summary.json')):
    seed = p.parent.name.replace('seed_', '')
    data = json.loads(p.read_text())
    for role, metrics in data.items():
        rows.append({'variant': 'gate', 'seed': seed, 'role': role,
                     **{k: metrics.get(k, float('nan')) for k in keys}})
out = root / 'summary_seeds.csv'
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
    if rows:
        w.writeheader()
        w.writerows(rows)


def mean(key, role='All'):
    vals = [float(r[key]) for r in rows
            if r['role'] == role and math.isfinite(float(r[key]))]
    return statistics.fmean(vals) if vals else float('nan')


print(f'wrote {out} ({len(rows)} rows)')
for role in ('Seen', 'Unseen', 'Novel', 'All'):
    if not any(r['role'] == role for r in rows):
        continue
    print(
        f'{role:6s} n={int(mean("n_datasets", role)):2d} '
        f'Dice {mean("coarse_dice", role):.4f}->{mean("prune_dice", role):.4f} '
        f'IoU {mean("coarse_iou", role):.4f}->{mean("prune_iou", role):.4f} '
        f'clDice {mean("coarse_cldice", role):.4f}->{mean("prune_cldice", role):.4f} '
        f'HD95 {mean("coarse_hd95", role):.2f}->{mean("prune_hd95", role):.2f} '
        f'gate AUC/AP {mean("gate_auc", role):.3f}/{mean("gate_auprc", role):.3f} '
        f'F1@.5 {mean("gate_f1_0_5", role):.3f} '
        f'bestF1 {mean("gate_best_f1", role):.3f} '
        f'topo AUC/AP {mean("gate_auc_topo", role):.3f}/{mean("gate_auprc_topo", role):.3f}')
PY

echo
echo "============================================================"
echo " All-20 re-evaluation done"
echo " summary: $OUT_ROOT/summary_seeds.csv"
echo " logs:    $OUT_ROOT/logs/"
echo "============================================================"
