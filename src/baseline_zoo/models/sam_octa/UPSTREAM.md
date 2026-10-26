# SAM-OCTA

- Upstream: https://github.com/ShellRedia/SAM-OCTA
- Vendored snapshot: `3rd/SAM-OCTA`, HEAD
  `9edc56247a3e30c96e5c6f1e8fe01250b8338527`.
- License: no LICENSE file in the vendored snapshot; add the upstream
  license before redistributing this copy.
- Files copied:
  - `segment_anything/` -> `segment_anything/` (absolute
    `segment_anything.*` imports rewritten as package-relative imports)
  - `sam_lora_image_encoder.py` -> `lora.py` (unused
    `SamPredictor`/`icecream` imports removed)
  - `loss_functions.py` -> `losses.py` (unchanged)
  - `prompt_points.py` -> `prompts.py` (unused `display` import removed)
- Modifications:
  - Added an optional `grad_checkpointing` switch to `ImageEncoderViT`
    (default off) for 1024x1024 LoRA training on 11 GiB.
  - ViT-L is used as the base (paper's shared SAM backbone); upstream
    defaults to ViT-H.
  - General datasets are fed as RGB, or grayscale replicated to 3 channels
    for OCTA-500, instead of the official three OCTA projection layers.
- Runtime consumer: `baseline_zoo.adapters.sam_octa.SAMOCTAAdapter`
