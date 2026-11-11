#!/usr/bin/env bash
# Overnight SSH hyperparameter screen (300 epochs, seed 0, local GPUs).
#
# Runs successive rounds through scripts/run_sacm_stc_ablation.sh; each
# round blocks until both arms finish, then the next round starts.  After
# the rounds, a summary CSV/Markdown compares every arm against the default
# w010 under the pre-registered rule:
#   eligible if last-10 Dice >= w010 + 0.3pp,
#               last-10 clDice >= w010 - 0.2pp,
#               last-10 HD95  <= w010 + 0.5;
#   otherwise keep the default 0.1.
#
# Usage (repo root):
#   bash scripts/run_sacm_stc_hparam_overnight.sh
#
# Env overrides:
#   GPU_LIST=0,1        EXTRA_ROUND=1   add the fusion20 round
#   DRY_RUN=1           print/validate commands, no training
#   PYTHON=...          interpreter (server/local default selection)
#   TMPDIR=...          default /home/dev/tmp (avoid the /tmp quota)
#   LOG_DIR=...         driver log directory
set -u

cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
CONFIG="configs/ablations/sacm_stc_ssh_hparam_300.yaml"
GPU_LIST="${GPU_LIST:-0,1}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ROUND="${EXTRA_ROUND:-0}"

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
export TMPDIR="${TMPDIR:-/home/dev/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$REPO/results/logs/sacm_stc_hparam_overnight_${STAMP}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1

echo "============================================================"
echo " SSH hyperparameter screen (300 epochs, seed 0)"
echo " config: $CONFIG"
echo " gpus:   $GPU_LIST"
echo " rounds: w005+w020, then w010+fusion05"
echo " extra:  ${EXTRA_ROUND} (1 adds fusion20)"
echo " log:    $LOG_DIR/driver.log"
echo "============================================================"

run_round() {
  local name="$1"
  shift
  local args=(--stage train --config "$CONFIG" --arms "$@" \
              --seeds 0 --gpus "$GPU_LIST")
  if [ "$DRY_RUN" = "1" ]; then
    args+=(--dry-run)
  fi
  echo ""
  echo "[round $name] $(date -Is) arms: $*"
  bash scripts/run_sacm_stc_ablation.sh "${args[@]}"
  local rc=$?
  echo "[round $name] exit=$rc $(date -Is)"
  return 0
}

run_round r1_w005_w020 w005 w020
run_round r2_w010_fusion05 w010 fusion05
if [ "$EXTRA_ROUND" = "1" ]; then
  run_round r3_fusion20 fusion20
fi

echo ""
echo "[summary] $(date -Is)"
"$PYTHON" - "$REPO" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
run_root = repo / 'results/runs/sacm_stc_ssh_hparam_300'
out_dir = repo / 'results/ablations/sacm_stc_ssh_hparam_300'
out_dir.mkdir(parents=True, exist_ok=True)
arms = ('w010', 'w005', 'w020', 'fusion05', 'fusion20')
metrics = ('dice', 'iou', 'cldice', 'hd95')
rows = []
for arm in arms:
    path = run_root / arm / 'seed_0' / 'metrics_history.json'
    if not path.is_file():
        continue
    history = json.loads(path.read_text())
    hist = history.get('history') or []
    if not hist:
        continue
    rows_map = {entry['epoch']: entry for entry in hist}
    best_epoch = int(history['best_epoch'])
    best = rows_map[best_epoch]['overall']
    tail = hist[-10:]
    row = {
        'arm': arm,
        'last_epoch': history['last_epoch'],
        'best_epoch': best_epoch,
        'best_dice': best['dice'],
        **{f'best_{m}': best[m] for m in metrics if m != 'dice'},
        **{f'last10_{m}': statistics.mean(
            entry['overall'][m] for entry in tail) for m in metrics},
    }
    rows.append(row)

reference = next((row for row in rows if row['arm'] == 'w010'), None)
eligible = []
for row in rows:
    row['eligible'] = ''
    row['reason'] = ''
    if reference is None:
        continue
    dd = row['last10_dice'] - reference['last10_dice']
    dc = row['last10_cldice'] - reference['last10_cldice']
    dh = row['last10_hd95'] - reference['last10_hd95']
    row['delta_dice'] = dd
    row['delta_cldice'] = dc
    row['delta_hd95'] = dh
    ok = dd >= 0.003 and dc >= -0.002 and dh <= 0.5
    row['eligible'] = 'yes' if (ok and row['arm'] != 'w010') else 'no'
    reasons = []
    if dd < 0.003:
        reasons.append(f'dice {dd:+.4f} < +0.003')
    if dc < -0.002:
        reasons.append(f'cldice {dc:+.4f} < -0.002')
    if dh > 0.5:
        reasons.append(f'hd95 {dh:+.2f} > +0.5')
    row['reason'] = '; '.join(reasons) if reasons else 'passes constraints'
    if row['eligible'] == 'yes':
        eligible.append(row)

columns = ['arm', 'last_epoch', 'best_epoch', 'best_dice', 'best_iou',
           'best_cldice', 'best_hd95', 'last10_dice', 'last10_iou',
           'last10_cldice', 'last10_hd95', 'delta_dice', 'delta_cldice',
           'delta_hd95', 'eligible', 'reason']
with (out_dir / 'summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)

lines = ['# SSH 300-epoch hyperparameter screen (seed 0)', '',
         'All arms share the same 300-epoch schedule and differ only in the '
         'SSH hyperparameter. The winner must be confirmed at 800 epochs.',
         '', '| arm | best | Dice | IoU | clDice | HD95 | last10 Dice | '
         'delta Dice | delta clDice | delta HD95 | eligible |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
for row in rows:
    lines.append(
        '| {arm} | E{best_epoch} | {best_dice:.4f} | {best_iou:.4f} | '
        '{best_cldice:.4f} | {best_hd95:.2f} | {last10_dice:.4f} | '
        '{dd:+.4f} | {dc:+.4f} | {dh:+.2f} | {eligible} |'.format(
            **row,
            dd=row.get('delta_dice', float('nan')),
            dc=row.get('delta_cldice', float('nan')),
            dh=row.get('delta_hd95', float('nan'))))
lines += ['', 'Decision rule: eligible if last10 Dice >= w010 + 0.003, '
          'last10 clDice >= w010 - 0.002, last10 HD95 <= w010 + 0.5; '
          'otherwise keep w010.']
if eligible:
    winner = max(eligible, key=lambda item: item['last10_dice'])
    lines += ['', f'Screen winner: **{winner["arm"]}** '
              f'(last10 Dice {winner["last10_dice"]:.4f}, '
              f'delta Dice {winner["delta_dice"]:+.4f}).'
              ' Confirm it at 800 epochs against the control.']
else:
    lines += ['', 'No candidate passed the constraints; keep the default w010.']
(out_dir / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'\nsummary: {out_dir / "summary.csv"}')
PY

echo "[done] $(date -Is)"
