#!/usr/bin/env bash
# Regenerate the standard dataset source pools (not the 3-shot split).
#
# Usage:
#   conda activate tsai
#   bash scripts/datasets/prepare_all.sh
#   bash scripts/datasets/prepare_all.sh DRIVE HRF TopoMortar
#
# Each prepare_<dataset>.py writes images/, masks/ and manifest.csv to its
# default active role path.  This runner passes --overwrite so a previously
# generated pool can be rebuilt.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
CONDA_ENV="${CONDA_ENV:-tsai}"
if [ -n "${PYTHON:-}" ]; then
  PY_CMD=("$PYTHON")
elif [ -x /home/dev/anaconda3/envs/"$CONDA_ENV"/bin/python ]; then
  PY_CMD=(/home/dev/anaconda3/envs/"$CONDA_ENV"/bin/python)
elif command -v conda >/dev/null 2>&1; then
  PY_CMD=(conda run -n "$CONDA_ENV" python)
else
  PY_CMD=(python)
fi
echo "Python: ${PY_CMD[*]}"
"${PY_CMD[@]}" -c 'import PIL, numpy, h5py; print("environment OK")' || {
  echo "ERROR: ${CONDA_ENV} environment is missing PIL/numpy/h5py"
  exit 1
}

ALL=(
  DRIVE CHASEDB1 DCA1 CREMI cracktree200
  FIVES STARE XCAD DeepCrack HRF LES-AV CHCrack5k
  ROAD tyre wire
  TopoMortar OCTA-500 DeepGlobe
)
if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=("${ALL[@]}")
fi

failed=0
run=0
for ds in "${DATASETS[@]}"; do
  script="scripts/datasets/prepare_${ds}.py"
  run=$((run + 1))
  echo
  echo "════════════════════════════════════════════════════════════"
  echo "[$run/${#DATASETS[@]}] $ds"
  echo "════════════════════════════════════════════════════════════"
  if [ ! -f "$script" ]; then
    echo "FAIL: missing script $script"
    failed=$((failed + 1))
    continue
  fi
  "${PY_CMD[@]}" "$script" --overwrite
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "OK: $ds"
  else
    echo "FAIL: $ds (exit $rc)"
    failed=$((failed + 1))
  fi
done

echo
echo "════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "════════════════════════════════════════════════════════════"
"${PY_CMD[@]}" - <<'PY'
from pathlib import Path
root = Path('datasets')
total_i = total_m = 0
for role in ('seen', 'unseen', 'novel'):
    base = root / role
    if not base.is_dir():
        continue
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        ni = len([p for p in (d / 'images').glob('*') if p.is_file()]) if (d / 'images').is_dir() else 0
        nm = len([p for p in (d / 'masks').glob('*') if p.is_file()]) if (d / 'masks').is_dir() else 0
        nf = len([p for p in (d / 'fov').glob('*') if p.is_file()]) if (d / 'fov').is_dir() else 0
        total_i += ni
        total_m += nm
        print(f'{role:6s} {d.name:16s} images={ni:5d} masks={nm:5d} fov={nf:3d}')
print(f'TOTAL images={total_i} masks={total_m}')
PY
summary_rc=$?
if [ "$summary_rc" -ne 0 ]; then
  echo "WARN: summary failed (exit $summary_rc)"
  failed=$((failed + 1))
fi

echo
if [ "$failed" -ne 0 ]; then
  echo "RESULT: $failed failure(s)"
  exit 1
fi
echo "RESULT: all ${#DATASETS[@]} dataset script(s) succeeded"
