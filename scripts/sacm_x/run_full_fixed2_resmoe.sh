#!/usr/bin/env bash
# full_fixed2_aux + Topology-aware residual MoE on fixed head 2.
#
# FineSAM++-style learned gate + deterministic residual experts.  All SACM
# components, C2F, multi-depth, aux heads and Skeleton Recall are unchanged.
#
# Usage: bash scripts/sacm_x/run_full_fixed2_resmoe.sh [smoke|train|full]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# Isolate the residual-MoE attribution from SCNP.
export SCNP=false
exec bash scripts/sacm_x/run_pepa_variant.sh full_fixed2_resmoe "${1:-full}"
