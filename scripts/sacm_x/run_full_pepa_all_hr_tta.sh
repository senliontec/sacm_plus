#!/usr/bin/env bash
# P4 deployment run: all PEPA/topology/ranker/high-res components + flip TTA.
# TTA is inference-side only and should be reported separately from the
# zero-TTA main table.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PRESET=full_pepa_all_hr
export INPUT_MODE=resize
export EVAL_INPUT_MODE=resize
export USE_TTA=true
exec bash scripts/sacm_x/run_pepa_variant.sh full_pepa_all_hr "${1:-full}"
