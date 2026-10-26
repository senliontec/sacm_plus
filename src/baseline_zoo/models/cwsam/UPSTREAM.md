# CWSAM (ClassWise-SAM-Adapter)

- Upstream: https://github.com/xypu98/CWSAM; the model derives from
  SAM-adapter.
- Vendored snapshot: `3rd/CWSAM`, HEAD
  `7bbb32ad3769ffc3f41f4361a074c55a7299fa27`.
- License: no LICENSE file in the vendored snapshot; add the upstream
  license before redistributing this copy.
- Files copied:
  - `models/sam.py` -> `sam.py` (registry import/decorator removed)
  - `models/iou_loss.py` -> `iou_loss.py` (unchanged)
  - `models/mmseg/models/common.py` -> `common.py` (unchanged)
  - `models/mmseg/models/sam/image_encoder.py` -> `image_encoder.py`
    (unchanged; the vendored reference already contains the optional
    `grad_checkpointing` branch)
  - `models/mmseg/models/sam/mask_decoder.py` -> `mask_decoder.py`
  - `models/mmseg/models/sam/transformer.py` -> `transformer.py`
- Modifications:
  - Removed the mmseg package/`mmcv` import chain; the copied modules use
    local relative imports.
  - The adapter enables the reference's per-block activation-checkpointing
    switch at runtime for 11 GiB training; the copied source is unchanged.
  - Instantiated at ViT-L scale per the paper's shared-backbone statement;
    the upstream SAR config uses ViT-B.
- Runtime consumer: `baseline_zoo.adapters.cwsam.CWSAMAdapter`
