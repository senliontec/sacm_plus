#!/usr/bin/env bash
# full_fixed2_aux + SCNP (training-only same-class neighbor penalty).
#
# Usage: bash scripts/sacm_x/run_full_fixed2_scnp.sh [smoke|train|full]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export SCNP=true
exec bash scripts/sacm_x/run_pepa_variant.sh full_fixed2_scnp "${1:-full}"
