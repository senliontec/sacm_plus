#!/usr/bin/env bash
# full_fixed2_aux + SCNP prevention + topology-aware ResMoE repair.
#
# Usage: bash scripts/sacm_x/run_full_fixed2_scnp_resmoe.sh [smoke|train|full]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export SCNP=true
exec bash scripts/sacm_x/run_pepa_variant.sh full_fixed2_scnp_resmoe "${1:-full}"
