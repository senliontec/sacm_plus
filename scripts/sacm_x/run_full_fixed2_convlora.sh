#!/usr/bin/env bash
# Control experiment: full_fixed2_aux + ConvLoRA only (no EVP, no DetailStem).
#
# Usage: bash scripts/sacm_x/run_full_fixed2_convlora.sh [smoke|train|full]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
exec bash scripts/sacm_x/run_pepa_variant.sh full_fixed2_convlora "${1:-full}"
