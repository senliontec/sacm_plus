# MaskVSC

- Upstream: https://github.com/Zhouyi-Zura/MaskVSC
- Vendored snapshot: `3rd/MaskVSC`, HEAD
  `5e60565b5ec63cb91295e06b9b502d6350e89d76`.
- License: the tree contains `LICENSE.txt` (Apache-2.0), but the README says
  "All rights reserved. The code and data are released for academic research
  use only." This conflict must be resolved before redistribution.
- Files copied:
  - `model/UNet.py` -> `unet.py` (unchanged)
  - `lib/Loss.py` -> `loss.py` (only `.cuda()` -> `img.device`)
  - `ImageDataset_Fundus` corruption branch -> `transforms.py`
    (extracted verbatim, plus an empty-skeleton guard)
  - `model/gcn.py` -> `gcn.py` (ViG completion head), with two minimal
    repairs described below
- Added wrapper: `full.py` (`MaskVSCFull`, stage-1 UNet + stage-2 ViG)
- Runtime consumer: `baseline_zoo.adapters.maskvsc.MaskVSCAdapter`

## Stage-2 repairs

The released stage-2 cannot run as shipped:

1. `ViGSeg_Fundus.forward` reshapes the `[B, N, C]` token tensor with
   `.view(B, N, patch, patch)`, which treats the channel vector as a spatial
   patch instead of reshaping the `N = (img/patch)^2` token grid. The port
   uses `transpose(1,2).reshape(B, C, side, side)`.
2. The released Fundus call passes `patch_size=16`, but the hardcoded
   predictor expects 1024 channels (=32^2); OCTA expects 576 (=24^2) and
   2PFM 1024. The port uses patch size 32 for the 512 Fundus variant and
   makes the first predictor conv use `patch_size^2`, which equals the
   upstream hardcoded width at that intended size.

No other ViG layer definitions or math were changed.

## Two-stage runtime

- Stage 1: unchanged official UNet + masked-vessel corruption + BCE +
  Connect_Loss, trained for 600 epochs by default.
- Stage 2: stage 1 is frozen (`requires_grad=False`) and forced to `eval()`;
  ViG is trained with BCE for 200 epochs on clean training images and the
  frozen stage-1 probability maps.
- Inference: `ViG(Seg(image))`, then threshold and resize to GT resolution.
- Stage-2 selection uses the 6-image validation Dice split. Upstream evaluates
  the test split every epoch and selects by test Dice; that is not used.
- A single pooled 512 ViG is used across domains. Upstream trains separate
  Fundus/OCTA/2PFM completion models per dataset; unseen/novel protocol
  domains have no per-domain training data, so a pooled model is the only
  protocol-consistent adaptation.
- The 800-epoch shared budget is split 600 stage-1 + 200 stage-2 by default.
  `--maskvsc-stage1-epochs 800` runs the ViG-isolation A/B (1000 total).
- The custom two-stage loop does not implement `--resume`, gradient
  accumulation, or gradient clipping.
