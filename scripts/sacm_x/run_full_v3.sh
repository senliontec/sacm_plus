#!/usr/bin/env bash
# SACM-X v3: dense structural decoder.
#   one external adapter, no prompt decoder, no candidate heads,
#   learned 64->512 upsampling, centerline/boundary/width aux heads,
#   image-guided 1024 refinement, Topograph + Skeleton Recall.
#
# Usage:
#   bash scripts/sacm_x/run_full_v3.sh smoke
#   bash scripts/sacm_x/run_full_v3.sh
#   bash scripts/sacm_x/run_full_v3.sh train
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
exec bash scripts/sacm_x/run_pepa_variant.sh full_v3 "${1:-full}"
