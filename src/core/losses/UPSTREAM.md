# Skeleton Recall loss — upstream and adaptation

- **Upstream project**: MIC-DKFZ/Skeleton-Recall
  <https://github.com/MIC-DKFZ/Skeleton-Recall>
- **Paper**: Kirchhoff et al., *Skeleton Recall Loss for Connectivity
  Conserving and Resource Efficient Segmentation of Thin Tubular
  Structures*, ECCV 2024.
- **License**: Apache License 2.0 (upstream repository).
- **Local file**: `skeleton_recall.py`

## What was reused

The soft-recall objective and the tubed-skeleton construction follow the
official implementation:

- `SoftSkeletonRecallLoss.forward` in
  `skeleton_recall/training/loss/dice.py`;
- `SkeletonTransform` in
  `skeleton_recall/training/data_augmentation/skeletonization.py`
  (`skimage.morphology.skeletonize`, 2-px cross dilation, clip to the
  segmentation).

## Deliberate adaptations

1. The official pipeline appends the skeleton as a second segmentation
   channel during data loading.  This project leaves its shared benchmark
   transforms untouched and computes the skeleton online from the same
   augmented GT used by the mask losses.
2. The loss returns `1 - recall` (positive, minimized) instead of the
   upstream negated recall, matching this project's clDice convention.
3. The default smoothing is `1.0` (this project's clDice convention) rather
   than the upstream `1e-5`; the formula is unchanged.
4. The nnU-Net DDP/batch-Dice all-gather path is not ported; the local
   trainer already averages per-sample losses and synchronizes gradients
   through DDP.

The upstream tree is used only as a read-only reference and is never
imported at runtime.
