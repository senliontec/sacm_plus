#!/usr/bin/env bash
# full_fixed2_aux + disagreement-gated residual experts (no learned gate).
#
# Usage: bash scripts/sacm_x/run_full_fixed2_resmoe_dg.sh [smoke|train|full]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# Isolate the disagreement-gated residual experts from SCNP.
export SCNP=false
exec bash scripts/sacm_x/run_pepa_variant.sh full_fixed2_resmoe_dg "${1:-full}"
