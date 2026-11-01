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

`UNetAdapter` keeps the official milesial architecture, BCE + official
global Dice, per-image scale preprocessing and logit-space bilinear
upsampling.  The 3-shot recipe follows the standard 2D supervised U-Net
trainer in Mamba-UNet/SSL4MIS (fixed-size inputs, large batch, long
schedule, flips/rotation, Dice+CE), adapted to native-resolution
curvilinear data: 512x512 random patches, half centred on a foreground
pixel, batch 8, 8 patches per training image per epoch, AdamW lr 1e-3/
wd 1e-4, cosine over 800 epochs, flips/90-degree rotations + mild
photometric jitter, gradient clipping 1.0.

The literal official 5-epoch RMSprop lr 1e-5/`ReduceLROnPlateau` recipe is
not used: on this split its first epoch (0.081 val Dice) became the plateau
best, and after three groups of five non-improving epochs the LR reached
1e-8; PyTorch's default `eps=1e-8` then prevented further reductions,
leaving the model nearly frozen from epoch 19.  Label the row "3-shot U-Net
(official architecture; adapted recipe)", not an official U-Net
reproduction.
