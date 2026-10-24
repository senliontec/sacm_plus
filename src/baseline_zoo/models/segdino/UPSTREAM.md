# SegDINO

- Upstream: vendored `baselines/segdino` (URL not recorded in the README)
- License: MIT in the vendored snapshot (`LICENSE`, copied verbatim).
- Files copied: `dpt.py`, `blocks.py` (relative `blocks` import).
- Modifications:
  - DPT accepts timm's DINOv3 backbone through `forward_intermediates`
    (the official hub API uses `get_intermediate_layers`); the DPT/DPTHead
    math is unchanged.
  - The official code expects a local DINOv3 GitHub checkout and gated
    DINOv3 weights. This port uses timm's `vit_*_patch16_dinov3` backbone
    and accepts `--dino-ckpt` / HF authentication.
- Runtime consumer: `baseline_zoo.adapters.segdino.SegDINOAdapter`
