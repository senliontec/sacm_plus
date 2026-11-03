#!/usr/bin/env bash
# P4 geometry run: PEPA + high-res DetailStem with native-crop training.
# Keeps evaluation at resize for comparability with the existing table.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PRESET=full_pepa_hr
export INPUT_MODE=native_crop
export EVAL_INPUT_MODE=resize
exec bash scripts/sacm_x/run_pepa_variant.sh full_pepa_hr "${1:-full}"
