# SAM-Med2D

- Upstream: https://github.com/OpenGVLab/SAM-Med2D
- Vendored snapshot: `3rd/SAM-Med2D`, HEAD
  `bfd2b93b1158100c8abd81f61766a2de92c1c175`.
- License: Apache-2.0 (`LICENSE`, copied verbatim)
- Files copied:
  - `segment_anything/` -> `segment_anything/` (absolute imports made
    package-relative)
  - `utils.py` -> `utils.py` (prompt generation, transforms, losses)
  - `LICENSE` -> `LICENSE`
- Modifications:
  - Package-relative imports only; no Apex import (the adapter never
    imports upstream `train.py`).
  - Default initialization is the official SAM-Med2D ViT-B checkpoint
    (`sam-med2d_b.pth`) at 256 input with `encoder_adapter=True`, matching
    the released test.py. A vanilla SAM ViT-L/H initialization remains an
    explicit `--sam-med2d-model-type vit_l`/`vit_h` comparison only; their
    default checkpoints are `sam_vit_l_0b3195.pth`/`sam_vit_h_4b8939.pth`.
    The official `load_from` interpolation path is used when dimensions
    differ, and a checkpoint missing non-adapter encoder tensors is
    rejected instead of silently random-initializing the architecture.
  - Input stays at the official 256x256; the BGR + SAM-statistics
    preprocessing quirk is preserved.
  - The official checkpoint contains both `model` and `optimizer`; loading
    uses `weights_only=False` because PyTorch 2.6 otherwise rejects the
    optimizer object. The checkpoint is an explicitly trusted official
    file.
  - `load_from()` no longer excludes `mask_tokens`,
    `output_hypernetworks_mlps` or `iou_prediction_head`.  Those tensors
    are shape-compatible between vanilla SAM and the 256-input model, and
    the controlled "fine-tune from SAM" protocol requires inheriting the
    full SAM decoder.  Positional embeddings and rel-pos tables are
    interpolated by shape; incompatible tensors are skipped.
- Runtime consumer: `baseline_zoo.adapters.sam_med2d.SamMed2DAdapter`
