# U-Net

- Upstream: https://github.com/milesial/Pytorch-UNet
- Vendored snapshot: `baselines/unet` (development reference tree)
- License: GPL-3.0 (`LICENSE`, copied verbatim)
- Files copied: `unet/__init__.py`, `unet/unet_model.py`,
  `unet/unet_parts.py`
- Modifications: none. Imports were already relative, so the package is
  self-contained.
- Runtime consumer: `baseline_zoo.adapters.unet.UNetAdapter`
