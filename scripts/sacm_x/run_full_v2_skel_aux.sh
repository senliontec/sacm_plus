#!/usr/bin/env bash
# Hybrid SACM-X: pretrained prompt decoder + training-only structural
# heads (centerline / boundary / width) + Skeleton Recall at 512.
#
# Usage:
#   bash scripts/sacm_x/run_full_v2_skel_aux.sh smoke
#   bash scripts/sacm_x/run_full_v2_skel_aux.sh
#   bash scripts/sacm_x/run_full_v2_skel_aux.sh train
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
exec bash scripts/sacm_x/run_pepa_variant.sh full_v2_skel_aux "${1:-full}"
