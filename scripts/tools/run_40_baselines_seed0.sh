#!/usr/bin/env bash
# One-click: run all 41 baselines (excluding SACM-X) for seed 0 on 8 GPUs.
# (The historical filename keeps the 40-baseline command working; K-Prism is
# now the 41st model.)
#
# Default: train + validation only.  Set FULL=1 for train -> test -> report
# -> figures.  Set KILL_GPU=1 to clear the selected GPUs first.
#
# Usage:
#   bash scripts/tools/run_40_baselines_seed0.sh
#   KILL_GPU=1 bash scripts/tools/run_40_baselines_seed0.sh
#   FULL=1 bash scripts/tools/run_40_baselines_seed0.sh
#
# Optional environment overrides:
#   GPUS=0,1,2,3,4,5,6,7
#   SEED=0
#   PYTHON=/home/zkyd/anaconda3/envs/tsai/bin/python
#   WANDB_MODE=online|offline|disabled
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODELS='a2sam,aco_moe_lora,bcunet,blo_sam,c2p,convlora,cs2net,csunet,cwsam,dconnnet,dka,dscnet,flowsdf,fssam,jtfn,knowsam,kprism,liot,maskvsc,matcher,medcagd,nnunet2d,nnwnet,persam,rsrm,s2mnet,sacm_official,sam2_unet,sam_adapter,sam_med2d,sam_octa,sam_vit_l_unet,samed,sansa,scrwkv,segdino,segdino_v2,topolora,tpsegformer,ucs,unet'
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SEED="${SEED:-0}"
KILL_GPU="${KILL_GPU:-0}"
FULL="${FULL:-0}"
DRY_RUN="${DRY_RUN:-0}"

count="$(printf '%s' "$MODELS" | awk -F, '{print NF}')"
echo "============================================================"
echo "41-baseline seed${SEED} run"
echo "models: ${count}"
echo "gpus:   ${GPUS}"
echo "mode:   $([ "$FULL" -eq 1 ] && echo 'full train+test+report' || echo 'train+validation only')"
echo "============================================================"

if [ "$count" -ne 41 ]; then
  echo "ERROR: expected 41 baselines, got ${count}" >&2
  exit 2
fi

if [ "$KILL_GPU" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
  bash scripts/tools/clear_gpus.sh --gpus "$GPUS"
fi

args=(--model "$MODELS" --seeds "$SEED" --gpus "$GPUS")
if [ "$DRY_RUN" -eq 1 ]; then
  args+=(--dry-run)
fi

if [ "$FULL" -eq 1 ]; then
  if [ "$KILL_GPU" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
    args+=(--kill-gpu)
  fi
  exec bash scripts/run_paper.sh "${args[@]}"
fi
exec bash scripts/run_train.sh "${args[@]}"
