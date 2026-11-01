#!/usr/bin/env bash
# One-click SACM+ (fixed official SACM) train + full eval.
#
# This intentionally delegates to the same wrappers as every other baseline
# (scripts/run_train.sh / scripts/run_eval.sh).  Those wrappers:
#   * select the tsai Python environment,
#   * source scripts/sacm_x/wandb_env.sh (self-hosted W&B, online),
#   * refuse to start when W&B is enabled but no credentials exist,
# so this script never triggers wandb's interactive login prompt.
#
# Usage:
#   GPU=0 bash scripts/tools/run_sacm_plus_seed0.sh
#   GPU=0 SEED=1 bash scripts/tools/run_sacm_plus_seed0.sh
#   GPU=0 FULL=1 bash scripts/tools/run_sacm_plus_seed0.sh   # + tables/figures
#
# W&B is online by default against the self-hosted server.  Do not set
# WANDB_MODE=offline unless that is explicitly intended.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
SEED="${SEED:-0}"
FULL="${FULL:-0}"

if [ "$FULL" -eq 1 ]; then
  exec bash scripts/run_paper.sh --model sacm_plus --seeds "$SEED" --gpus "$GPU"
fi

bash scripts/run_train.sh --model sacm_plus --seeds "$SEED" --gpus "$GPU"
bash scripts/run_eval.sh --model sacm_plus --seeds "$SEED" --gpus "$GPU"
