#!/usr/bin/env bash
# Recommended SACM-X v2:
#   external-only adapters + DetailStem 512/1024 + TADT
#   + single main head + Topograph/Skeleton Recall.
#
# Usage:
#   bash scripts/sacm_x/run_full_v2.sh
#   bash scripts/sacm_x/run_full_v2.sh smoke
#   bash scripts/sacm_x/run_full_v2.sh train
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
exec bash scripts/sacm_x/run_pepa_variant.sh full_v2 "${1:-full}"
