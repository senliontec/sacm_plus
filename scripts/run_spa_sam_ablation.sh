#!/usr/bin/env bash
# One-click SPA-SAM ablation driver.
#
# Each variant gets its own run root, W&B group and summary.  The driver calls
# scripts/run_spa_sam_seeds03_800.sh so training, fixed3 derivation, 20-dataset
# evaluation and W&B logging follow exactly the same path as the main model.
#
# Usage:
#   bash scripts/run_spa_sam_ablation.sh
#   EPOCHS=200 VARIANTS="aug_none aug_mild aug_strong" \
#       bash scripts/run_spa_sam_ablation.sh
#   DRY_RUN=1 bash scripts/run_spa_sam_ablation.sh
#
# Env (passed through to the main driver):
#   SEEDS, EPOCHS, TRAIN_GPUS, EVAL_GPUS, RESUME, SKIP_TRAIN, SKIP_EVAL,
#   FORCE_DERIVE, FORCE_EVAL, DRY_RUN, USE_WANDB, WANDB_*
#   VARIANTS="aug_none aug_flip aug_photo aug_mild aug_strong
#             no_specialized no_fusion no_topology dense_selected
#             topo_deployment"
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

VARIANTS_STR="${VARIANTS:-aug_none aug_flip aug_photo aug_mild aug_strong no_specialized no_fusion no_topology dense_selected topo_deployment}"
VARIANTS_STR="${VARIANTS_STR//,/ }"
read -r -a VARIANT_ARGS <<< "$VARIANTS_STR"
if [ "${#VARIANT_ARGS[@]}" -eq 0 ]; then
  echo "ERROR: VARIANTS is empty" >&2
  exit 2
fi

for variant in "${VARIANT_ARGS[@]}"; do
  case "$variant" in
    aug_none)        aug=none;  extra="" ;;
    aug_flip)        aug=flip;  extra="" ;;
    aug_photo)       aug=photo; extra="" ;;
    aug_mild)        aug=mild;  extra="" ;;
    aug_strong)      aug=strong; extra="" ;;
    no_specialized)  aug=mild;  extra="--specialized-heads 0" ;;
    no_fusion)       aug=mild;  extra="--head-fusion 0" ;;
    no_topology)     aug=mild;  extra="--cl-dice-weight 0 --skeleton-recall-weight 0 --specialized-topology-weight 0" ;;
    dense_selected)  aug=mild;  extra="--stage2-loss-scope selected" ;;
    topo_deployment) aug=mild;  extra="--topology-head deployment" ;;
    *)
      echo "ERROR: unknown SPA-SAM ablation variant: $variant" >&2
      exit 2
      ;;
  esac
  echo "============================================================"
  echo " SPA-SAM ablation: ${variant}  train-aug=${aug}  extra=${extra:-none}"
  echo "============================================================"
  TRAIN_AUG="$aug" \
  EXTRA_ARGS="$extra" \
  RUN_ROOT="results/runs/spa_sam_ablation/${variant}" \
  SUMMARY_ROOT="results/ablations/spa_sam_ablation/${variant}" \
  WANDB_GROUP="spa_sam_ablation_${variant}" \
  WANDB_PREFIX="spa-sam-${variant}" \
  bash scripts/run_spa_sam_seeds03_800.sh
done

echo "All requested SPA-SAM ablation variants finished."
