#!/usr/bin/env bash
# One command for the full overnight comparison on 8 GPUs.
#
#   GPU 0-4: final model, data seeds 0-4, 800 epochs
#            full_fixed2 + gate + support-calibrated component prune
#   GPU 5-7: SACM official, data seeds 0-2, 800 epochs
#            training + all-20-dataset test
#
# Both pipelines run concurrently; W&B is enabled (the server has it).
# NO_AUGMENT=1 by default: SACM official's recipe has no train-time
# augmentation, so the final model is trained on the same clean inputs.
# After both finish, this script writes a compact final-vs-official summary.
#
# Usage:
#   bash scripts/run_all_tonight.sh
#   DRY_RUN=1 bash scripts/run_all_tonight.sh
#   FINAL_SEEDS="0 1 2 3 4" OFFICIAL_SEEDS="0 1 2" bash scripts/run_all_tonight.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi

FINAL_SEEDS="${FINAL_SEEDS:-0 1 2 3 4}"
OFFICIAL_SEEDS="${OFFICIAL_SEEDS:-0 1 2}"
FINAL_GPUS="${FINAL_GPUS:-0,1,2,3,4}"
OFFICIAL_GPUS="${OFFICIAL_GPUS:-5,6,7}"
EPOCHS="${EPOCHS:-800}"
NO_AUGMENT="${NO_AUGMENT:-1}"
DRY_RUN="${DRY_RUN:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-results/runs/tonight_all_${STAMP}}"
FINAL_OUT="$OUT_ROOT/final"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$FINAL_OUT" "$LOG_DIR"

for seed in $FINAL_SEEDS; do
  root="data/sacm_3shot_seed${seed}"
  if [ ! -f "$root/protocol_manifest.csv" ]; then
    echo "ERROR: missing $root/protocol_manifest.csv" >&2
    echo "The final model needs data seeds: $FINAL_SEEDS" >&2
    exit 2
  fi
done
for seed in $OFFICIAL_SEEDS; do
  root="data/sacm_3shot_seed${seed}"
  if [ ! -f "$root/protocol_manifest.csv" ]; then
    echo "ERROR: missing $root/protocol_manifest.csv" >&2
    exit 2
  fi
done
if [ ! -f checkpoints/sam_vit_l_0b3195.pth ]; then
  echo "ERROR: missing checkpoints/sam_vit_l_0b3195.pth" >&2
  exit 2
fi

echo "============================================================"
echo " overnight all-in-one"
echo " final:    GPU $FINAL_GPUS | data seeds $FINAL_SEEDS | $EPOCHS epochs"
echo " official: GPU $OFFICIAL_GPUS | data seeds $OFFICIAL_SEEDS | $EPOCHS epochs"
echo " output:   $OUT_ROOT"
echo " augment:  final=$([ "$NO_AUGMENT" = "1" ] && echo disabled || echo enabled) | official=disabled (official recipe)"
echo " wandb:    online"
echo "============================================================"

if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] final -> scripts/sacm_x/run_component_full.sh"
  echo "[dry-run] official -> scripts/tools/run_sacm_official_5to7.sh"
  exit 0
fi

# The final pipeline itself manages GPU0-4: one seed per GPU.
# PER_GPU_DATA_SEED=1 aligns its splits with the official baseline.
(
  export OUT_ROOT="$FINAL_OUT"
  export TRAIN_SEEDS="$FINAL_SEEDS"
  export PER_GPU_DATA_SEED=1
  export EPOCHS="$EPOCHS"
  export RUN_NO_GATE=0
  export USE_WANDB=true
  export NO_AUGMENT="$NO_AUGMENT"
  bash scripts/sacm_x/run_component_full.sh
) > "$LOG_DIR/final_pipeline.log" 2>&1 &
final_pid=$!

# SACM official uses GPU5-7 and runs its own 20-dataset test after training.
(
  GPU_LIST="$OFFICIAL_GPUS" SEEDS="$OFFICIAL_SEEDS" \
    bash scripts/tools/run_sacm_official_5to7.sh
) > "$LOG_DIR/official_pipeline.log" 2>&1 &
official_pid=$!

echo "final PID:    $final_pid"
echo "official PID: $official_pid"
echo "logs: $LOG_DIR"
echo "waiting for both pipelines..."

set +e
wait "$final_pid"; final_rc=$?
wait "$official_pid"; official_rc=$?
set -e

echo "final rc=$final_rc, official rc=$official_rc"

# Compact comparison.  final has data seeds 0-4; official has 0-2.
"$PYTHON" - "$OUT_ROOT" "$FINAL_OUT" "$OFFICIAL_SEEDS" "$final_rc" "$official_rc" <<'PY'
import csv, json, math, statistics, sys
from pathlib import Path

out_root = Path(sys.argv[1])
final_out = Path(sys.argv[2])
official_seeds = sys.argv[3].split()
final_rc = int(sys.argv[4])
official_rc = int(sys.argv[5])


def mean(vals):
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else float('nan')


final_rows = []
p = final_out / 'summary_seeds.csv'
if p.is_file():
    with p.open(newline='') as f:
        final_rows = list(csv.DictReader(f))

official = []
for seed in official_seeds:
    q = Path(f'results/runs/sacm_official/default/seed_{seed}/eval/tables/macro_metrics.csv')
    if not q.is_file():
        continue
    with q.open(newline='') as f:
        for row in csv.DictReader(f):
            if row.get('role') != 'All':
                continue
            try:
                official.append({'seed': seed, 'metric': row.get('metric'),
                                 'mean': float(row.get('mean'))})
            except (TypeError, ValueError):
                pass

FINAL_KEYS = ('coarse_dice', 'prune_dice',
              'coarse_iou', 'prune_iou',
              'coarse_cldice', 'prune_cldice',
              'coarse_hd95', 'prune_hd95',
              'coarse_components', 'prune_components',
              'coarse_cc_error', 'prune_cc_error')


def aggregate(rows):
    return {key: mean([r.get(key) for r in rows]) for key in FINAL_KEYS}


all_final_seeds = sorted({r['seed'] for r in final_rows})
matched_seeds = sorted(set(all_final_seeds) & set(official_seeds))
summary = {
    'final_rc': final_rc,
    'official_rc': official_rc,
    'final_seeds': all_final_seeds,
    'official_seeds': official_seeds,
    'final_vs_official_seeds': matched_seeds,
    # Headline comparison: only seeds present in BOTH runs.
    'final': {},
    # Full final run (may include extra seeds the official baseline lacks).
    'final_all': {},
    'official': {},
}
if final_rows:
    gate = [r for r in final_rows
            if r.get('variant') == 'gate' and r.get('role') == 'All']
    summary['final_all'] = aggregate(gate)
    summary['final'] = aggregate(
        [r for r in gate if r['seed'] in matched_seeds])
for metric in ('dice', 'cldice', 'hd95', 'iou'):
    vals = [r['mean'] for r in official if r['metric'] == metric]
    if vals:
        summary['official'][metric] = mean(vals)

(out_root / 'comparison_summary.json').write_text(
    json.dumps(summary, indent=2) + '\n')
print(f"headline final seeds: {matched_seeds} | "
      f"final_all seeds: {all_final_seeds}")
print(json.dumps(summary, indent=2))
PY

if [ "$final_rc" != "0" ] || [ "$official_rc" != "0" ]; then
  echo "One or more pipelines failed. Inspect $LOG_DIR" >&2
  exit 6
fi

echo "============================================================"
echo " ALL DONE"
echo " final:    $FINAL_OUT"
echo " official: results/runs/sacm_official/default/seed_{0,1,2}/eval/tables/"
echo " compare:  $OUT_ROOT/comparison_summary.json"
echo " logs:     $LOG_DIR"
echo "============================================================"
