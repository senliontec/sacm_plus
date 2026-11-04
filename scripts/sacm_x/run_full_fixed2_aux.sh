#!/usr/bin/env bash
# full/fixed2 + structural supervision (256 decoder).
#   keeps 4 heads / FusionV2 / multi-depth / C2F; fixed head 2.
#
# Usage: bash scripts/sacm_x/run_full_fixed2_aux.sh [smoke|train|full]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
exec bash scripts/sacm_x/run_pepa_variant.sh full_fixed2_aux "${1:-full}"
