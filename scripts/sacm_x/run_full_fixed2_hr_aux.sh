#!/usr/bin/env bash
# full/fixed2 + structural supervision + DetailStem 512 / 1024 refine.
# Needs ~24GB (RTX 3090); use the 8x3090 server.
#
# Usage: bash scripts/sacm_x/run_full_fixed2_hr_aux.sh [smoke|train|full]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
exec bash scripts/sacm_x/run_pepa_variant.sh full_fixed2_hr_aux "${1:-full}"
