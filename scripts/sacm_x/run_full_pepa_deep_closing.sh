#!/usr/bin/env bash
# P1 diagnostic: PEPA checkpoint + Deep Closing post-processing.
# Deep Closing is model-agnostic; this wrapper only fixes PRESET=full_pepa.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PRESET=full_pepa
exec bash scripts/sacm_x/run_sacm_x_deep_closing.sh
