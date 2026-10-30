# U-Net

- Upstream: https://github.com/milesial/Pytorch-UNet
- Vendored snapshot: `3rd/Pytorch-UNet`, HEAD
  `21d7850f2af30a9695bbeea75f3136aa538cfc4a`.
- License: GPL-3.0 (`LICENSE`, copied verbatim)
- Files copied: `unet/__init__.py`, `unet/unet_model.py`,
  `unet/unet_parts.py`
- Modifications: none. Imports were already relative, so the package is
  self-contained.
- Runtime consumer: `baseline_zoo.adapters.unet.UNetAdapter`

## Adapter recipe

`UNetAdapter` follows the official `train.py`/`predict.py` recipe: per-image
`scale=0.5` (variable input size), batch 1, RMSprop (lr 1e-5, wd 1e-8,
momentum 0.999), `ReduceLROnPlateau('max', patience=5)` on validation Dice,
no augmentation, BCE + official global Dice, gradient clipping 1.0, and
logit-space bilinear upsampling to the original size.

The shared 3-shot protocol supplies 18 train / 6 val images, extends the
official 5-epoch default to 800 epochs, uses the 6-image validation split
for checkpoint selection, and keeps the official 1-class binary head
(official `--classes 1` branch).  These protocol adaptations are listed in
`UNetAdapter.deviations`.
