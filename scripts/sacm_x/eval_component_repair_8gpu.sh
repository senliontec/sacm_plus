#!/usr/bin/env bash
# One-click 8-GPU component-repair evaluation.
#
# Usage:
#   bash scripts/sacm_x/eval_component_repair_8gpu.sh
#   MODE=old bash scripts/sacm_x/eval_component_repair_8gpu.sh
#   MODE=verify LIMIT=2 bash scripts/sacm_x/eval_component_repair_8gpu.sh
#   MODE=verify DROP_GATE_FEATURES=1 bash scripts/sacm_x/eval_component_repair_8gpu.sh
#
# MODE=verify (default):
#   GPU 0-4: full_fixed2_aux seeds 0-4
#   GPU 5:   full_fixed2_scnp seed 0
#   GPU 6:   full_fixed2_resmoe seed 0
#   GPU 7:   full_fixed2_resmoe_dg seed 0
#
# MODE=old:
#   the previous 8-model encoder screen (aux/evp/convlora/lora_evp/peft).
#
# The component classifier is trained per checkpoint on seen support images
# (leave-one-dataset-out for Seen targets) and thresholds are support-calibrated.
# Unseen/Novel use the global seen-support threshold.
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

MODE="${MODE:-verify}"
DATA_ROOT="${DATA_ROOT:-data/sacm_3shot_seed0}"
TEST_ROOT="${TEST_ROOT:-$DATA_ROOT/test}"
LIMIT="${LIMIT:-0}"
DROP_GATE_FEATURES="${DROP_GATE_FEATURES:-0}"
FORCE="${FORCE:-0}"
OUT_ROOT="${OUT_ROOT:-results/component_repair_8gpu/$MODE}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

case "$MODE" in
  verify)
    SCREEN_ROOT="${SCREEN_ROOT:-results/runs/cvpr8gpu_verify}"
    JOB_SPECS=(
      "0 full_fixed2_aux 0"
      "1 full_fixed2_aux 1"
      "2 full_fixed2_aux 2"
      "3 full_fixed2_aux 3"
      "4 full_fixed2_aux 4"
      "5 full_fixed2_scnp 0"
      "6 full_fixed2_resmoe 0"
      "7 full_fixed2_resmoe_dg 0"
    )
    ;;
  old)
    SCREEN_ROOT="${SCREEN_ROOT:-results/runs/cvpr8gpu_screen}"
    JOB_SPECS=(
      "0 full_fixed2_aux 0"
      "1 full_fixed2_evp 0"
      "2 full_fixed2_convlora 0"
      "3 full_fixed2_lora_evp 0"
      "4 full_fixed2_peft_lora_evp 0"
      "5 full_fixed2_aux 1"
      "6 full_fixed2_evp 1"
      "7 full_fixed2_lora_evp 1"
    )
    ;;
  *)
    echo "ERROR: MODE must be verify|old, got $MODE" >&2
    exit 2
    ;;
esac

resolve_ckpt() {
  local preset="$1" seed="$2" cand
  for cand in \
    "$SCREEN_ROOT/sacm_x/$preset/seed_$seed/$preset/best_model.pth" \
    "$SCREEN_ROOT/sacm_x/$preset/seed_$seed/best_model.pth" \
    "$SCREEN_ROOT/$preset/seed_$seed/best_model.pth" \
    "$SCREEN_ROOT/$preset/seed_$seed/$preset/best_model.pth"; do
    [ -f "$cand" ] && { printf '%s\n' "$cand"; return 0; }
  done
  cand="$(find "$SCREEN_ROOT" \( -path "*/$preset/seed_$seed/*best_model.pth" \
    -o -path "*/$preset/seed_$seed/best_model.pth" \) 2>/dev/null | head -1)"
  [ -n "$cand" ] && { printf '%s\n' "$cand"; return 0; }
  return 1
}

echo "============================================================"
echo " component repair eval: $MODE"
echo " screen: $SCREEN_ROOT"
echo " data:   $DATA_ROOT"
echo " test:   $TEST_ROOT"
echo " out:    $OUT_ROOT"
echo " limit:  $LIMIT (0 = full)"
echo " gate features: $([ "$DROP_GATE_FEATURES" = "1" ] && echo dropped || echo auto)"
echo " python: $PY"
echo "============================================================"

declare -a PIDS=()
declare -a NAMES=()
for spec in "${JOB_SPECS[@]}"; do
  read -r gpu preset seed <<< "$spec"
  ckpt="$(resolve_ckpt "$preset" "$seed")" || {
    echo "MISSING checkpoint: $preset seed $seed under $SCREEN_ROOT" >&2
    exit 3
  }
  run_out="$OUT_ROOT/$preset/seed_${seed}"
  log="$LOG_DIR/${preset}_seed${seed}.log"
  if [ "$FORCE" != "1" ] && [ -f "$run_out/summary.json" ]; then
    echo "[gpu $gpu] SKIP existing $run_out"
    continue
  fi
  mkdir -p "$run_out"
  extra=()
  [ "$DROP_GATE_FEATURES" = "1" ] && extra+=(--drop-gate-features)
  echo "[gpu $gpu] $preset seed $seed -> $run_out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/sacm_x/component_repair_eval.py \
    --preset "$preset" \
    --checkpoint "$ckpt" \
    --data-root "$DATA_ROOT" \
    --test-root "$TEST_ROOT" \
    --output-dir "$run_out" \
    --device cuda:0 \
    --limit "$LIMIT" \
    --run-name "${preset}_seed${seed}" \
    "${extra[@]}" > "$log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${preset}_seed${seed}")
done

fail=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "DONE ${NAMES[$i]}"
  else
    echo "FAILED ${NAMES[$i]} (log: $LOG_DIR/${NAMES[$i]}.log)" >&2
    fail=1
  fi
done

if [ "$fail" != "0" ]; then
  echo "Some jobs failed; inspect logs under $LOG_DIR" >&2
  exit 4
fi

"$PY" - "$OUT_ROOT" <<'PY'
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for p in sorted(root.glob('*/seed_*/summary.json')):
    d = json.loads(p.read_text())
    run = p.parent.relative_to(root).as_posix()
    m = d.get('All')
    if not m:
        continue
    rows.append({'run': run, **{k: m.get(k, float('nan')) for k in (
        'n_datasets', 'coarse_dice', 'prune_dice',
        'coarse_cldice', 'prune_cldice',
        'coarse_hd95', 'prune_hd95',
        'coarse_components', 'prune_components',
        'coarse_cc_error', 'prune_cc_error')}})
out = root / 'summary_all.csv'
with out.open('w', newline='') as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print(f'wrote {out} ({len(rows)} runs)')
if rows:
    print('run'.ljust(48), 'Dice', 'clDice', 'HD95', 'cc_err')
    for r in rows:
        print(r['run'].ljust(48),
              f"{r['prune_dice']:.4f} ({r['prune_dice']-r['coarse_dice']:+.4f})",
              f"{r['prune_cldice']:.4f} ({r['prune_cldice']-r['coarse_cldice']:+.4f})",
              f"{r['prune_hd95']:.2f} ({r['prune_hd95']-r['coarse_hd95']:+.2f})",
              f"{r['prune_cc_error']:.1f} ({r['prune_cc_error']-r['coarse_cc_error']:+.1f})")
PY

echo "All done. Results: $OUT_ROOT"
