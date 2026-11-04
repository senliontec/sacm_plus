#!/usr/bin/env bash
# Replacement-style PEFT ablation: Conv-LoRA/EVP replace the entire SACM
# adapter stack (no internal/external adapters, no FusionV2).  Use this to
# isolate how much of the gain comes from the SACM adapters.
#
# Usage: bash scripts/sacm_x/run_full_fixed2_peft_lora_evp.sh [smoke|train|full]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
exec bash scripts/sacm_x/run_pepa_variant.sh full_fixed2_peft_lora_evp "${1:-full}"
