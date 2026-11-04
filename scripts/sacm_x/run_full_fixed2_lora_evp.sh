#!/usr/bin/env bash
# Hybrid SACM + Conv-LoRA/EVP on top of the fixed2 decoder.
#   keeps SACM external geometric adapters + FusionV2, drops the duplicate
#   block-internal adapter, and adds official-style Conv-LoRA qkv residuals
#   plus EVP FFT prompts.  Fixed head 2, C2F, multi-depth, aux@512 and
#   Skeleton Recall are retained.
#
# Usage: bash scripts/sacm_x/run_full_fixed2_lora_evp.sh [smoke|train|full]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
exec bash scripts/sacm_x/run_pepa_variant.sh full_fixed2_lora_evp "${1:-full}"
