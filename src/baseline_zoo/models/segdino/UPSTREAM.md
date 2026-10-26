# SegDINO

- Upstream: https://github.com/script-Yang/segdino
- Vendored snapshot: `3rd/segdino`, HEAD
  `c078ca0e2c7578304e80eaa4e9d3f7c6dd0f3c41`.
- License: MIT in the vendored snapshot (`LICENSE`, copied verbatim).
- Files copied: `dpt.py`, `blocks.py` (relative `blocks` import).
- Modifications:
  - DPT accepts timm's DINOv3 backbone through `forward_intermediates`
    (the official hub API uses `get_intermediate_layers`); the DPT/DPTHead
    math is unchanged.
  - The official code expects a local DINOv3 GitHub checkout and gated
    DINOv3 weights. This port uses timm's `vit_*_patch16_dinov3` backbone,
    auto-detects a size-matched local checkpoint under `checkpoints/`,
    accepts `--dino-ckpt` / `DINO_CKPT` / `.safetensors`, and falls back
    to HF authentication only when no local file exists.
- Runtime consumer: `baseline_zoo.adapters.segdino.SegDINOAdapter`
