# FoB-SAM (CVPR 2026)

- Upstream: `baselines/FoB_SAM` (vendored snapshot)
- Paper: *Focus on the Background: Exploring SAM's Potential in Few-shot
  Medical Image Segmentation with Background-centric Prompting*, CVPR 2026
- License: Apache License 2.0 (copied as `LICENSE`)
- Files copied:
  - `models/FoB.py` -> `model.py`
  - `models/encoder.py` -> `encoder.py`
  - `SAM.py` -> `sam.py`
  - `dataloaders/image_transforms.py` -> `transforms.py`
- Runtime consumer: `baseline_zoo.adapters.fob_sam.FoBSAMAdapter`

## Model logic preserved

- `FewShotSeg` / `IDR` / `SPG` / `SPR` / `Head` / `PromptMatching` /
  `MaskedAttention` / `JointsMSELoss` are copied verbatim, including the
  official loss composition (`heatmap_loss*1000 + L2/10000 + rac_loss +
  foreground_loss`).
- Background prompt construction: contour sampling with the official
  21/15-pixel ring dilation, Gaussian keypoint heatmaps (sigma=4),
  masked-average prototypes, query attention suppression, and
  structure-guided prompt refinement (`SPR`, 3 iterations).
- Frozen SAM point-prompt inference is preserved.  SAM modeling code is
  reused from the already-vendored `baseline_zoo.models.sam_octa`
  Segment Anything copy (official Meta code), not duplicated.

## Deviations required by this repository

- The official model hard-codes CUDA in `FewShotSeg.__init__`,
  `MaskedAttention.__init__` and the NLL weight.  This port registers
  `sin_pos` and the NLL weight as buffers and makes the model
  device-agnostic so CPU smoke tests and `.to(device)` work.
- Absolute checkpoint paths are replaced by
  `--resnet-checkpoint` / `--sam-checkpoint`.
- `info_nce` is an optional dependency.  The official forward never
  calls `self.InfoNCE`; a dependency-free placeholder is provided when
  the package is absent.
- The official training is 3D-volume/episodic with supervoxel pseudo
  classes.  The 20-dataset adapter uses 2D support/query pairs with GT
  support masks (`use_gt=True` equivalent), 1-shot training episodes and
  same-domain validation support.
- For unseen/novel test datasets (no labeled support in the 3-shot
  split), the adapter uses a structurally matched donor support set:
  crack-like datasets -> `CrackTree200`, vessel-like datasets -> `DRIVE`;
  override with `--fob-support-dataset`.
- Prompt predictions and SAM masks are produced at 256x256 and resized
  to the original GT resolution for the shared metrics.
- The official test passes `query_labels` into `forward`; it is used only
  for the empty-foreground guard.  The adapter passes an all-ones dummy
  at evaluation, so test GT never influences prompts or masks.
- SAM `best_pred_idx=0` (the official non-ISIC setting); expose
  `--fob-best-pred-idx` for completeness.
