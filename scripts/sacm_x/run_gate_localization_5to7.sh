#!/usr/bin/env bash
# Evaluate gate localization on all 20 datasets using existing checkpoints.
#
# This is diagnostic-only: no training, no component pruning, no official
# baseline.  It uses GPU5-7 so it does not interfere with a final component
# evaluation running on GPU0-4.
#
# Usage (server):
#   bash scripts/sacm_x/run_gate_localization_5to7.sh
#
# Output:
#   <final_root>/gate_localization_all20_<timestamp>/seed_<s>/
#     per_dataset.csv
#     summary.json
#   <final_root>/gate_localization_all20_<timestamp>/summary_seeds.csv
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
# checkpoint lives under full_fixed2_gate; use GATE_PRESET for an explicit
# override.
PRESET="${GATE_PRESET:-full_fixed2_gate}"
SEEDS="${SEEDS:-0 1 2 3 4}"
GPU_LIST="${GPU_LIST:-5 6 7}"
LIMIT="${LIMIT:-0}"
FINAL_ROOT="${FINAL_ROOT:-}"
if [ -z "$FINAL_ROOT" ]; then
  FINAL_ROOT="$(ls -dt results/runs/final_gpu0_4_* 2>/dev/null | head -1 || true)"
fi
if [ -z "$FINAL_ROOT" ] || [ ! -d "$FINAL_ROOT" ]; then
  echo "ERROR: final run directory not found; set FINAL_ROOT=..." >&2
  exit 2
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$FINAL_ROOT/gate_localization_all20_${STAMP}}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

read -r -a SEED_ARGS <<< "$SEEDS"
read -r -a GPU_ARGS <<< "$GPU_LIST"
if [ "${#GPU_ARGS[@]}" -eq 0 ]; then
  echo "ERROR: GPU_LIST is empty" >&2
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
echo " Gate localization on 20 datasets"
echo " final:  $FINAL_ROOT"
echo " preset: $PRESET"
echo " seeds:  $SEEDS"
echo " gpus:   $GPU_LIST"
echo " out:    $OUT_ROOT"
echo " limit:  $LIMIT (0 = all images)"
echo "============================================================"

PIDS=()
NAMES=()
launch() {
  local seed="$1" gpu="$2"
  local ckpt data_root protocol log
  ckpt="$(resolve_ckpt "$seed")" || {
    echo "ERROR: missing checkpoint for seed $seed under $FINAL_ROOT" >&2
    return 1
  }
  data_root="data/sacm_3shot_seed${seed}"
  protocol="$data_root/protocol_manifest.csv"
  if [ ! -f "$protocol" ]; then
    echo "ERROR: missing protocol $protocol" >&2
    return 1
  fi
  log="$LOG_DIR/eval_seed${seed}.log"
  echo "[gpu $gpu] seed $seed: $ckpt"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" scripts/sacm_x/eval_gate_localization.py \
      --preset "$PRESET" \
      --checkpoint "$ckpt" \
      --data-root "$data_root" \
      --protocol "$protocol" \
      --expected-datasets 20 \
      --output-dir "$OUT_ROOT/seed_${seed}" \
      --device cuda:0 \
      --limit "$LIMIT" \
      --run-name "seed${seed}_gate"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("seed${seed}")
}

flush_batch() {
  local i rc=0
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      echo "GATE EVAL DONE ${NAMES[$i]}"
    else
      echo "GATE EVAL FAILED ${NAMES[$i]} (log: $LOG_DIR/eval_${NAMES[$i]}.log)" >&2
      rc=1
    fi
  done
  PIDS=()
  NAMES=()
  return "$rc"
}

idx=0
batch_fail=0
for seed in "${SEED_ARGS[@]}"; do
  gpu="${GPU_ARGS[$((idx % ${#GPU_ARGS[@]}))]}"
  launch "$seed" "$gpu" || { batch_fail=1; break; }
  idx=$((idx + 1))
  if [ $((idx % ${#GPU_ARGS[@]})) -eq 0 ]; then
    flush_batch || batch_fail=1
    [ "$batch_fail" = "0" ] || break
  fi
done
if [ "${#PIDS[@]}" -gt 0 ]; then
  flush_batch || batch_fail=1
fi
if [ "$batch_fail" != "0" ]; then
  echo "Stopping: gate evaluation failures." >&2
  exit 4
fi

"$PY" - "$OUT_ROOT" <<'PY'
import csv, json, math, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
keys = (
    'gate_auc', 'gate_auprc', 'gate_f1_0_5',
    'gate_precision_0_5', 'gate_recall_0_5', 'gate_iou_0_5',
    'gate_best_f1', 'gate_best_iou', 'gate_best_threshold',
    'gate_topo_auc', 'gate_topo_auprc', 'gate_topo_f1_0_5',
)
rows = []
for p in sorted(root.glob('seed_*/summary.json')):
    seed = p.parent.name.replace('seed_', '')
    data = json.loads(p.read_text())
    for role, metrics in data.items():
        rows.append({'seed': seed, 'role': role, 'n_datasets':
                     metrics.get('n_datasets', float('nan')),
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
        f'AUROC {mean("gate_auc", role):.4f} '
        f'AUPRC {mean("gate_auprc", role):.4f} '
        f'F1@.5 {mean("gate_f1_0_5", role):.4f} '
        f'P/R@.5 {mean("gate_precision_0_5", role):.3f}/'
        f'{mean("gate_recall_0_5", role):.3f} '
        f'IoU@.5 {mean("gate_iou_0_5", role):.3f} '
        f'bestF1 {mean("gate_best_f1", role):.3f} '
        f'topo AUC/AP {mean("gate_topo_auc", role):.3f}/'
        f'{mean("gate_topo_auprc", role):.3f}')
PY

echo
echo "============================================================"
echo " Gate localization done"
echo " summary: $OUT_ROOT/summary_seeds.csv"
echo " logs:    $OUT_ROOT/logs/"
echo "============================================================"
