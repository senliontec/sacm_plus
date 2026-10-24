#!/usr/bin/env bash
# Generate seed-controlled 3-shot SACM splits.
#
# Usage:
#   bash scripts/data/make_shot3_split.sh                 # seeds 0 1 2 3 4
#   bash scripts/data/make_shot3_split.sh 7 8             # explicit seeds
#   SEEDS="0 1 2 3 4" bash scripts/data/make_shot3_split.sh
#   CLEAN=true bash scripts/data/make_shot3_split.sh 0 1 2 3 4
#
# Environment:
#   OUT_PREFIX           output prefix (default: data/sacm_3shot_seed)
#   CLEAN=true           rebuild existing seed directories (default: false)
#   SAMPLE_WITHIN_GROUP  shuffle CREMI volume slices (default: true)
#
# Each seed produces:
#   <OUT_PREFIX><seed>/train/{images,masks}
#   <OUT_PREFIX><seed>/val/{images,masks}
#   <OUT_PREFIX><seed>/test/<dataset>/{images,masks}
#   <OUT_PREFIX><seed>/split_manifest.csv
#   <OUT_PREFIX><seed>/protocol_manifest.csv
#   <OUT_PREFIX><seed>/protocol_summary.json
#   <OUT_PREFIX><seed>/seed_config.json
#
# CREMI has only three volumes.  Volume roles stay fixed as
# train=B, val=A, test=C; SAMPLE_WITHIN_GROUP only changes which slices
# inside B/A are used, so the Seen test volume never moves.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

if [ -z "${PYTHON:-}" ]; then
  if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/dev/anaconda3/envs/tsai/bin/python
  elif [ -x /home/zkyd/anaconda3/envs/tsai/bin/python ]; then
    PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
  else
    PYTHON=python
  fi
fi

if [ "$#" -gt 0 ]; then
  SEEDS="$*"
else
  SEEDS="${SEEDS:-0 1 2 3 4}"
fi

OUT_PREFIX="${OUT_PREFIX:-data/sacm_3shot_seed}"
CLEAN="${CLEAN:-false}"
SAMPLE_WITHIN_GROUP="${SAMPLE_WITHIN_GROUP:-true}"

TRAIN_DIRS=(
  datasets/seen/DRIVE
  datasets/seen/CHASEDB1
  datasets/seen/DCA1
  datasets/seen/CREMI
  datasets/seen/CrackTree200
  CORN:datasets/seen/CORN/train
)
VAL_DIRS=(CORN:datasets/seen/CORN/val)
TEST_DIRS=(CORN:datasets/seen/CORN/test)

case "$CLEAN" in
  true|1|yes) CLEAN_ARG=(--clean) ;;
  false|0|no) CLEAN_ARG=() ;;
  *) echo "ERROR: CLEAN must be true/false, got '$CLEAN'" >&2; exit 2 ;;
esac
case "$SAMPLE_WITHIN_GROUP" in
  true|1|yes) SAMPLE_ARG=(--sample_within_group) ;;
  false|0|no)
    SAMPLE_ARG=()
    echo "WARNING: SAMPLE_WITHIN_GROUP=false; CREMI slices will not change by seed." >&2
    ;;
  *) echo "ERROR: SAMPLE_WITHIN_GROUP must be true/false, got '$SAMPLE_WITHIN_GROUP'" >&2
     exit 2 ;;
esac

for seed in $SEEDS; do
  case "$seed" in
    ''|*[!0-9]*) echo "ERROR: seed must be a non-negative integer, got '$seed'" >&2; exit 2 ;;
  esac
  out="${OUT_PREFIX}${seed}"
  if [ "$out" = "data/sacm_3shot" ]; then
    echo "ERROR: refusing to overwrite the frozen data/sacm_3shot split" >&2
    exit 2
  fi
  if [ -e "$out" ] && [ "$CLEAN" != "true" ]; then
    echo "ERROR: $out already exists; re-run with CLEAN=true to rebuild it." >&2
    exit 3
  fi

  echo
  echo "=== seed $seed -> $out ==="
  mkdir -p "$out"
  "$PYTHON" scripts/data/prepare_data.py \
    --train_dirs "${TRAIN_DIRS[@]}" \
    --val_dirs "${VAL_DIRS[@]}" \
    --test_dirs "${TEST_DIRS[@]}" \
    --shots 3 --val_shots 1 --make_test \
    "${SAMPLE_ARG[@]}" "${CLEAN_ARG[@]}" \
    --out_dir "$out" --seed "$seed" 2>&1 | tee "$out/prepare.log"

  "$PYTHON" scripts/data/build_protocol_manifest.py \
    --split-root "$out" --datasets-root datasets --seed "$seed" \
    2>&1 | tee "$out/protocol.log"

  "$PYTHON" - "$out" "$seed" "$SAMPLE_WITHIN_GROUP" <<'PY'
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
seed = int(sys.argv[2])
sample_within_group = sys.argv[3].lower() in ('true', '1', 'yes')
datasets = ['DRIVE', 'CHASEDB1', 'DCA1', 'CREMI', 'CrackTree200', 'CORN']

with open(root / 'split_manifest.csv', newline='') as f:
    rows = list(csv.DictReader(f))
by = defaultdict(list)
for row in rows:
    by[(row['dataset'], row['split'])].append(row)
counts = Counter(row['split'] for row in rows)

errors = []
if counts['train'] != 18:
    errors.append(f"train={counts['train']} (expected 18)")
if counts['val'] != 6:
    errors.append(f"val={counts['val']} (expected 6)")
if counts['test'] != 643:
    errors.append(f"seen test={counts['test']} (expected 643)")
for dataset in datasets:
    if len(by[(dataset, 'train')]) != 3:
        errors.append(f"{dataset}: train={len(by[(dataset, 'train')])} (expected 3)")
    if len(by[(dataset, 'val')]) != 1:
        errors.append(f"{dataset}: val={len(by[(dataset, 'val')])} (expected 1)")
    names = {
        split: {row['source_image'] for row in by[(dataset, split)]}
        for split in ('train', 'val', 'test')
    }
    if names['train'] & names['val'] or names['train'] & names['test']:
        errors.append(f"{dataset}: image overlap across train/val/test")

cremi_train_groups = {row['group'] for row in by[('CREMI', 'train')]}
cremi_val_groups = {row['group'] for row in by[('CREMI', 'val')]}
cremi_test = [row['source_image'] for row in by[('CREMI', 'test')]]
if cremi_train_groups != {'CREMI/B'}:
    errors.append(f"CREMI train groups={sorted(cremi_train_groups)}, expected B")
if cremi_val_groups != {'CREMI/A'}:
    errors.append(f"CREMI val groups={sorted(cremi_val_groups)}, expected A")
if not all(name.startswith('sample_C_') for name in cremi_test):
    errors.append("CREMI test is not entirely volume C")

summary = json.loads((root / 'protocol_summary.json').read_text())
if summary.get('seed') != seed:
    errors.append(f"protocol summary seed={summary.get('seed')} (expected {seed})")
if summary.get('totals', {}).get('rows') != 3604:
    errors.append(
        f"protocol rows={summary.get('totals', {}).get('rows')} (expected 3604)")
if not summary.get('group_checks', {}).get('ok'):
    errors.append('protocol group check failed')

if errors:
    print(f'[error] seed {seed} validation failed:', file=sys.stderr)
    for error in errors:
        print(f'  - {error}', file=sys.stderr)
    raise SystemExit(1)

config = {
    'seed': seed,
    'shots': 3,
    'val_shots': 1,
    'sample_within_group': sample_within_group,
    'protocol_version': summary.get('protocol_version'),
    'split_root': str(root),
    'protocol_manifest_sha256': summary.get('manifest_sha256'),
}
(root / 'seed_config.json').write_text(json.dumps(config, indent=2) + '\n')

print(f"seed {seed}: train={counts['train']} val={counts['val']} "
      f"test={counts['test']} manifest_rows={summary['totals']['rows']} "
      f"group_ok={summary['group_checks']['ok']}")
for dataset in datasets:
    train = sorted(row['source_image'] for row in by[(dataset, 'train')])
    val = sorted(row['source_image'] for row in by[(dataset, 'val')])
    print(f"  {dataset:13s} train={','.join(train)} | val={','.join(val)}")
PY
done

echo
echo "done: ${SEEDS}"
