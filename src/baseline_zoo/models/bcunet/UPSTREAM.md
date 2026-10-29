# BCUNet

- Upstream: https://github.com/xzhong411/BCU-Net
- Vendored snapshot: `3rd/BCU-Net`, HEAD
  `008ff9292097791a3f12c2776aea398f71b7c996`.
- License: no LICENSE file in the vendored snapshot; add the upstream
  license before redistributing this copy.
- Files copied:
  - `code/core/models.py` -> `model.py` (imports made relative)
  - `code/core/convnext.py` -> `convnext.py` (unchanged)
  - `code/core/unet_parts.py` -> `unet_parts.py`
- Modifications:
  - `OutConv` calls an undefined `upconv2x2`; repaired as a 1x1 logits
    projection at full resolution. The paper describes four normal
    upsampling blocks and sends the U-Net branch's final feature map to
    MRL fusion; it does not specify an extra upconv2x2/BN/sigmoid. This is
    a CE-compatible reconstruction, not official code.
  - No official BCU-Net trained weights are released or required for the
    from-scratch shared protocol; only the ConvNeXt ImageNet backbone is
    pretrained.
  - `warmlearnrate.py` is missing upstream and only used by `main.py`, which
    is not part of this copy.
- Runtime consumer: `baseline_zoo.adapters.bcunet.BCUNetAdapter`
