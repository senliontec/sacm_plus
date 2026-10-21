#!/usr/bin/env bash
set -euo pipefail

# Run from any working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# No arguments: export the default volume directly under data/.
# Any arguments are forwarded to extract_cremi.py unchanged. --src is
# prepended as a default; a later user-supplied --src overrides it.
#
# Examples:
#   bash scripts/extract_cremi.sh
#   bash scripts/extract_cremi.sh --hdf sample_C_20160501.hdf \
#       --out data --skip-blank
if [ "$#" -eq 0 ]; then
    set -- \
        --hdf sample_C_20160501.hdf \
        --out data \
        --skip-blank \
        --overwrite
fi

exec "${PYTHON:-python}" scripts/extract_cremi.py \
    --src ../datasets/CREMI \
    "$@"
