# BCUNet

- Upstream: URL not recorded in the vendored README
  (`baselines/bcunet/README.md`).
- Vendored snapshot: `baselines/bcunet` (development reference tree)
- License: no LICENSE file in the vendored snapshot; add the upstream
  license before redistributing this copy.
- Files copied:
  - `code/core/models.py` -> `model.py` (imports made relative)
  - `code/core/convnext.py` -> `convnext.py` (unchanged)
  - `code/core/unet_parts.py` -> `unet_parts.py`
- Modifications:
  - `OutConv` called an undefined `upconv2x2`; repaired as a 1x1 logits
    projection at full resolution (the U-Net branch is already full size).
  - `warmlearnrate.py` is missing upstream and only used by `main.py`, which
    is not part of this copy.
- Runtime consumer: `baseline_zoo.adapters.bcunet.BCUNetAdapter`
