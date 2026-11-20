#!/usr/bin/env bash
# One-click exact reproduction of the current best CoSNA result.
#
# It never trains. It copies canonical CoSNA checkpoints into a clean run
# root, evaluates all 20 datasets for seeds 0-3, and writes a 4-seed summary.
# No BCTD code or BCTD checkpoint path is used.
#
# Put the four canonical CoSNA checkpoints at:
#   checkpoints/cosna_best/seed_{0,1,2,3}/best_model.pth
# (or set COSNA_BEST_WEIGHTS to their parent directory).
#
# Usage:
#   bash scripts/reproduce_cosna_best.sh
#   DRY_RUN=1 bash scripts/reproduce_cosna_best.sh
#   SEEDS="0" GPUS="0" bash scripts/reproduce_cosna_best.sh
#
# Outputs:
#   results/runs/cosna/best_repro/seed_<n>/eval/tables/all_metrics.csv
#   results/reports/cosna_best_repro/summary.{json,csv,md}
#   results/reports/cosna_best_repro/seed_<n>_mean_std.{csv,md}
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

SEEDS_STR="${SEEDS:-0 1 2 3}"
SEEDS_STR="${SEEDS_STR//,/ }"
GPUS_STR="${GPUS:-0,1,2,3}"
RUN_ROOT="${RUN_ROOT:-results/runs/cosna/best_repro}"
SUMMARY_DIR="${SUMMARY_DIR:-results/reports/cosna_best_repro}"
USE_WANDB="${USE_WANDB:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-4000}"

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
export TMPDIR="${TMPDIR:-$REPO/results/.tmp}"
mkdir -p "$TMPDIR"

read -r -a SEED_ARGS <<< "$SEEDS_STR"
read -r -a GPU_ARGS <<< "${GPUS_STR//,/ }"
if [ "${#SEED_ARGS[@]}" -eq 0 ] || [ "${#GPU_ARGS[@]}" -eq 0 ]; then
  echo "ERROR: empty SEEDS/GPUS" >&2
  exit 2
fi

abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$REPO/$1" ;;
  esac
}

# Locate a directory containing seed_<n>/best_model.pth for every requested
# seed. Prefer CoSNA-owned assets; the bctd path is only a local historical
# location for canonical CoSNA weights, never a runtime code dependency.
CANDIDATES=()
if [ -n "${COSNA_BEST_WEIGHTS:-}" ]; then
  CANDIDATES+=("$(abspath "$COSNA_BEST_WEIGHTS")")
fi
CANDIDATES+=(
  "$REPO/checkpoints/cosna_best"
  "$REPO/checkpoints/cosna_init"
  "$REPO/results/runs/bctd/tcsu20"
)
WEIGHTS_ROOT=""
for candidate in "${CANDIDATES[@]}"; do
  [ -d "$candidate" ] || continue
  ok=1
  for seed in "${SEED_ARGS[@]}"; do
    [ -f "$candidate/seed_${seed}/best_model.pth" ] || { ok=0; break; }
  done
  if [ "$ok" = 1 ]; then
    WEIGHTS_ROOT="$candidate"
    break
  fi
done
if [ -z "$WEIGHTS_ROOT" ]; then
  cat >&2 <<'EOF'
ERROR: no canonical CoSNA checkpoints found.

The 62.09 champion is a trained weight artifact; it cannot be recreated by
code alone. Put four checkpoints on the server at:

  checkpoints/cosna_best/seed_0/best_model.pth
  checkpoints/cosna_best/seed_1/best_model.pth
  checkpoints/cosna_best/seed_2/best_model.pth
  checkpoints/cosna_best/seed_3/best_model.pth

or set COSNA_BEST_WEIGHTS=<directory containing seed_<n>/best_model.pth>.
Then rerun: bash scripts/reproduce_cosna_best.sh

If those weights are unavailable, the only independent alternative is a
scratch run (scripts/run_cosna_server_seeds03.sh); that is a different recipe
and will not reproduce 62.09.
EOF
  exit 3
fi

RUN_ROOT_ABS=$(abspath "$RUN_ROOT")
SUMMARY_DIR_ABS=$(abspath "$SUMMARY_DIR")
WEIGHTS_ROOT_ABS="$WEIGHTS_ROOT"
echo "CoSNA best reproduction"
echo "  weights: ${WEIGHTS_ROOT_ABS}"
echo "  seeds:   ${SEEDS_STR}"
echo "  gpus:    ${GPUS_STR}"
echo "  output:  ${RUN_ROOT}"
echo "  summary: ${SUMMARY_DIR}"

if [ "$SKIP_GPU_CHECK" != 1 ] && [ "$DRY_RUN" = 0 ]; then
  "$PYTHON" - "$GPUS_STR" "$MAX_USED_MIB" "$MIN_FREE_MIB" <<'PY'
import subprocess
import sys

gpus = [g.strip() for g in sys.argv[1].split(',') if g.strip()]
max_used = int(sys.argv[2])
min_free = int(sys.argv[3])
out = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free',
     '--format=csv,noheader,nounits'], text=True)
available = {}
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 4:
        continue
    try:
        available[parts[0]] = (parts[1], int(float(parts[2])),
                               int(float(parts[3])))
    except ValueError:
        continue
missing = [g for g in gpus if g not in available]
if missing:
    raise SystemExit(f'ERROR: GPUs not present: {missing}')
bad = []
for g in gpus:
    name, used, free = available[g]
    if used > max_used or free < min_free:
        bad.append((g, name, used, free))
if bad:
    for g, name, used, free in bad:
        print(f'ERROR: GPU {g} ({name}): used={used} MiB, '
              f'free={free} MiB', file=sys.stderr)
    raise SystemExit('free the GPUs or reduce the batch of requested GPUs')
print('GPU preflight OK')
PY
fi

env \
  MODE=repro \
  REPRO_ROOT="$WEIGHTS_ROOT_ABS" \
  RUN_ROOT="$RUN_ROOT" \
  FORCE=1 \
  SEEDS="$SEEDS_STR" \
  GPUS="$GPUS_STR" \
  USE_WANDB="$USE_WANDB" \
  DRY_RUN="$DRY_RUN" \
  PYTHON="$PYTHON" \
  bash "$REPO/scripts/run_cosna_baseline_seeds03.sh"

if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN=1: reproduction commands printed; nothing executed."
  exit 0
fi

"$PYTHON" "$REPO/scripts/pipeline/summarize_baseline_seeds.py" \
  --run-root "$RUN_ROOT_ABS" \
  --seeds "${SEED_ARGS[@]}" \
  --protocol "$REPO/data/sacm_3shot_seed0/protocol_manifest.csv" \
  --out-dir "$SUMMARY_DIR_ABS"

variant="$(basename "$RUN_ROOT")"
if [ "$RUN_ROOT_ABS" = "$REPO/results/runs/cosna/$variant" ]; then
  "$PYTHON" "$REPO/scripts/pipeline/record_4metrics.py" \
    --model cosna --variant "$variant" \
    --seeds "${SEED_ARGS[@]}" --expected-datasets 20
fi

"$PYTHON" - "$SUMMARY_DIR_ABS/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
expected = {'dice': 62.0875, 'iou': 48.4175,
            'cldice': 70.2125, 'hd95': 59.066}
actual = summary['macro_pct']
print('\n4-seed macro:',
      ' '.join(f'{k}={actual[k]:.4f}' for k in ('dice', 'iou', 'cldice', 'hd95')))
for key, want in expected.items():
    delta = abs(float(actual[key]) - want)
    if delta > 0.05:
        print(f'WARNING: {key} differs from canonical {want:.4f} '
              f'by {delta:.4f}; verify the checkpoint set.')
PY

echo
echo "DONE."
echo "  per-seed tables: $RUN_ROOT/seed_*/eval/tables/all_metrics.csv"
echo "  4-seed summary:  $SUMMARY_DIR/summary.{json,csv,md}"
