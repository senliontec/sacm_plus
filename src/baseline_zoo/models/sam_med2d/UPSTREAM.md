# SAM-Med2D

- Upstream: https://github.com/OpenGVLab/SAM-Med2D
- Vendored snapshot: `baselines/sam-med2d` (development reference tree)
- License: Apache-2.0 (`LICENSE`, copied verbatim)
- Files copied:
  - `segment_anything/` -> `segment_anything/` (absolute imports made
    package-relative)
  - `utils.py` -> `utils.py` (prompt generation, transforms, losses)
  - `LICENSE` -> `LICENSE`
- Modifications:
  - Package-relative imports only; no Apex import (the adapter never
    imports upstream `train.py`).
  - Instantiated at ViT-L scale with vanilla SAM ViT-L weights (the official
    SAM-Med2D checkpoint is not present; the paper states all SAM baselines
    share the ViT-L encoder). The official `load_from` interpolation path is
    used for `image_size=256`.
  - Input stays at the official 256x256; the BGR + SAM-statistics
    preprocessing quirk is preserved.
- Runtime consumer: `baseline_zoo.adapters.sam_med2d.SamMed2DAdapter`
