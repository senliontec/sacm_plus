#!/usr/bin/env bash
# Finish the Phase-0 six-dataset audits in one command.
#
# The main orchestrator (run_phase0_audit.sh) passes --datasets unquoted, so
# audit_predictions.py receives only the first dataset name and argparse
# rejects the rest.  This script re-runs the five missing six-dataset audits
# with a quoted dataset list, then builds the head-selection comparison.
#
# Does NOT train, evaluate, or rerun the topology ranker.
#
# Usage:
#   bash scripts/sacm_x/finish_phase0_audits.sh
#
# Environment overrides:
#   AUDIT_ROOT  default results/selector_audit_clean
#   PROTOCOL    default data/sacm_3shot_seed0/protocol_manifest.csv
#   PYTHON      default python
#   WORKERS     default 8
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

PYTHON="${PYTHON:-python}"
AUDIT_ROOT="${AUDIT_ROOT:-$REPO/results/selector_audit_clean}"
PROTOCOL="${PROTOCOL:-$REPO/data/sacm_3shot_seed0/protocol_manifest.csv}"
WORKERS="${WORKERS:-8}"
SIX="CREMI LEAF OCTA-500 TYRE TopoMortar WIRE"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="$AUDIT_ROOT/logs_audits_$STAMP"
mkdir -p "$LOG_ROOT"

need_file() {
  [ -f "$1" ] || { echo "ERROR: missing $1" >&2; exit 2; }
}

need_dir() {
  [ -d "$1" ] || { echo "ERROR: missing $1" >&2; exit 2; }
}

pred_dir_for() {
  local root="$1"
  local pred="$root/canonical/sacm_x/full/seed_0/eval"
  [ -d "$pred" ] || pred="$root/eval/full"
  printf '%s\n' "$pred"
}

need_file "$PROTOCOL"
need_file "$REPO/scripts/pipeline/audit_predictions.py"
need_file "$REPO/scripts/pipeline/compare_audits.py"

F2_ROOT="$AUDIT_ROOT/full_fixed2_20/seed_0"
IOU_ROOT="$AUDIT_ROOT/full_iou_6/seed_0"
H0_ROOT="$AUDIT_ROOT/full_fixed0_6/seed_0"
H1_ROOT="$AUDIT_ROOT/full_fixed1_6/seed_0"
H3_ROOT="$AUDIT_ROOT/full_fixed3_6/seed_0"

F2_AUDIT="$AUDIT_ROOT/full_fixed2_6/audit"
IOU_AUDIT="$IOU_ROOT/audit"
H0_AUDIT="$H0_ROOT/audit"
H1_AUDIT="$H1_ROOT/audit"
H3_AUDIT="$H3_ROOT/audit"
COMPARE="$AUDIT_ROOT/head_selection_6"
RANKER_ROOT="${RANKER_ROOT:-$REPO/results/topo_ranker_candidates/full_fixed2_6/seed_0/ranker}"

run_audit() {
  local tag="$1" pred="$2" out="$3"
  need_dir "$pred"
  echo
  echo "=== audit: $tag ==="
  "$PYTHON" scripts/pipeline/audit_predictions.py \
    --pred-root "$pred" \
    --protocol "$PROTOCOL" \
    --out "$out" \
    --datasets "$SIX" \
    --workers "$WORKERS" \
    2>&1 | tee "$LOG_ROOT/audit_${tag}.log"
  need_file "$out/per_dataset.csv"
  need_file "$out/macro_summary.csv"
}

echo "repo:     $REPO"
echo "protocol: $PROTOCOL"
echo "audits:   $AUDIT_ROOT"
echo "logs:     $LOG_ROOT"

# Existing 20-dataset audits should have succeeded in the main run.  Report
# them instead of silently assuming the comparison inputs are complete.
for f in \
  "$F2_ROOT/audit/per_dataset.csv" \
  "$AUDIT_ROOT/full_fixed2_resize_20/audit/per_dataset.csv"; do
  if [ -f "$f" ]; then
    echo "20-dataset audit: OK $f"
  else
    echo "WARNING: missing 20-dataset audit: $f" >&2
  fi
done

run_audit fixed2_6 "$(pred_dir_for "$F2_ROOT")" "$F2_AUDIT"
run_audit iou_6 "$(pred_dir_for "$IOU_ROOT")" "$IOU_AUDIT"
run_audit head0_6 "$(pred_dir_for "$H0_ROOT")" "$H0_AUDIT"
run_audit head1_6 "$(pred_dir_for "$H1_ROOT")" "$H1_AUDIT"
run_audit head3_6 "$(pred_dir_for "$H3_ROOT")" "$H3_AUDIT"

echo
echo "=== head selection comparison ==="
"$PYTHON" scripts/pipeline/compare_audits.py \
  --audit iou="$IOU_AUDIT" \
  --audit fixed0="$H0_AUDIT" \
  --audit fixed1="$H1_AUDIT" \
  --audit fixed2="$F2_AUDIT" \
  --audit fixed3="$H3_AUDIT" \
  --primary fixed2 \
  --out "$COMPARE" \
  2>&1 | tee "$LOG_ROOT/compare_audits.log"

for f in \
  macro_comparison.csv \
  per_dataset_comparison.csv \
  comparison.md \
  summary.json; do
  need_file "$COMPARE/$f"
done

echo
echo "=== ranker outputs ==="
missing_rankers=()
for ds in $SIX; do
  if [ -f "$RANKER_ROOT/$ds/summary.json" ]; then
    echo "ranker $ds: OK"
  else
    echo "WARNING: ranker $ds missing: $RANKER_ROOT/$ds/summary.json" >&2
    missing_rankers+=("$ds")
  fi
done

echo
echo "=== done ==="
echo "head comparison: $COMPARE/comparison.md"
echo "comparison csv:  $COMPARE/macro_comparison.csv"
echo "logs:            $LOG_ROOT"
if [ "${#missing_rankers[@]}" -gt 0 ]; then
  echo "WARNING: missing ranker results: ${missing_rankers[*]}" >&2
fi
