#!/usr/bin/env bash
#
# Audit predicted masks against the original test GT for all three SACM
# variants, then build one combined comparison table.
#
# Overridable:
#   SEED=0
#   RUNS_ROOT=/home/dev/Downloads/runs
#   PROTOCOL=data/sacm_3shot_seed0/protocol_manifest.csv
#   SACM_X_VARIANT=full
#   AUDIT_ROOT=${RUNS_ROOT}/audit
#   WORKERS=8
#   OFFICIAL_SUFFIX=.png
#   PLUS_SUFFIX=.png
#   X_SUFFIX=_result.png
#
# Prediction roots match the canonical run layout:
#   baselines: <run>/predictions/<method>
#   SACM-X:    <run>/eval
#
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SEED="${SEED:-0}"
RUNS_ROOT="${RUNS_ROOT:-/home/dev/Downloads/runs}"
AUDIT_ROOT="${AUDIT_ROOT:-${RUNS_ROOT}/audit}"
PROTOCOL="${PROTOCOL:-data/sacm_3shot_seed${SEED}/protocol_manifest.csv}"
SACM_X_VARIANT="${SACM_X_VARIANT:-full}"
WORKERS="${WORKERS:-8}"
EMPTY_POLICY="${EMPTY_POLICY:-penalty}"
PYTHON="${PYTHON:-python}"
OFFICIAL_SUFFIX="${OFFICIAL_SUFFIX:-.png}"
PLUS_SUFFIX="${PLUS_SUFFIX:-.png}"
X_SUFFIX="${X_SUFFIX:-_result.png}"

OFFICIAL_PRED="${OFFICIAL_PRED:-${RUNS_ROOT}/sacm_official/default/seed_${SEED}/predictions/sacm_official}"
PLUS_PRED="${PLUS_PRED:-${RUNS_ROOT}/sacm_plus/default/seed_${SEED}/predictions/sacm_plus}"
X_PRED="${X_PRED:-${RUNS_ROOT}/sacm_x/${SACM_X_VARIANT}/seed_${SEED}/eval}"

OFFICIAL_OUT="${AUDIT_ROOT}/sacm_official_default_seed${SEED}"
PLUS_OUT="${AUDIT_ROOT}/sacm_plus_default_seed${SEED}"
X_OUT="${AUDIT_ROOT}/sacm_x_${SACM_X_VARIANT}_seed${SEED}"
COMBINED_OUT="${AUDIT_ROOT}/three_model_seed${SEED}"

if [[ ! -f "$PROTOCOL" ]]; then
    echo "missing protocol manifest: $PROTOCOL" >&2
    exit 1
fi

run_audit() {
    local name="$1"
    local pred_root="$2"
    local out_dir="$3"
    local pred_suffix="$4"

    if [[ ! -d "$pred_root" ]]; then
        echo "[$name] missing prediction root: $pred_root" >&2
        return 1
    fi

    echo
    echo "=== audit: $name ==="
    echo "pred-root: $pred_root"
    echo "out:       $out_dir"
    echo "suffix:    $pred_suffix"

    "$PYTHON" scripts/pipeline/audit_predictions.py \
        --pred-root "$pred_root" \
        --protocol "$PROTOCOL" \
        --out "$out_dir" \
        --pred-suffix "$pred_suffix" \
        --workers "$WORKERS" \
        --empty-policy "$EMPTY_POLICY"
}

# Baselines save predictions under the original image stem (e.g.
# CHASEDB1_Image_02L.png); SACM-X saves <stem>_result.png.
ok=1
run_audit sacm_official "$OFFICIAL_PRED" "$OFFICIAL_OUT" "$OFFICIAL_SUFFIX" || ok=0
run_audit sacm_plus "$PLUS_PRED" "$PLUS_OUT" "$PLUS_SUFFIX" || ok=0
run_audit sacm_x "$X_PRED" "$X_OUT" "$X_SUFFIX" || ok=0

if [[ "$ok" -ne 1 ]]; then
    echo
    echo "One or more audits failed; skipping the combined comparison." >&2
    exit 1
fi

echo
echo "=== combined comparison ==="
"$PYTHON" scripts/pipeline/compare_audits.py \
    --audit "sacm_official=${OFFICIAL_OUT}" \
    --audit "sacm_plus=${PLUS_OUT}" \
    --audit "sacm_x=${X_OUT}" \
    --primary sacm_x \
    --out "$COMBINED_OUT"

echo
echo "Done."
echo "  official audit: ${OFFICIAL_OUT}/report.md"
echo "  plus audit:     ${PLUS_OUT}/report.md"
echo "  x audit:        ${X_OUT}/report.md"
echo "  comparison:     ${COMBINED_OUT}/comparison.md"