# SCRWKV upstream record

## Sources

| Component | Upstream | Commit |
|---|---|---|
| Model / training / CUDA WKV | `zhxhzy/SCRWKV`, `models/{SFE,SCIU,AMCM,CSHF,DySample,decoder}.py`, `models/cuda_new/`, `mmcls/SFE_dev/models/{SFE,modules/patch_embed}.py`, `engine.py`, `main.py`, `test.py` | `f3a1063f6043c769ef828c2ae95edf1029d76c40` |

Paper: "SCRWKV: Ultra-Compact Structure-Calibrated Vision-RWKV for
Topological Crack Segmentation", ICML 2026, arXiv:2605.14926.

License: the checkout has no LICENSE file; the README badge says Apache-2.0.
Verify redistribution terms with the authors before releasing.

## Copied / adapted

- `AMCM.py`, `CSHF.py`, `DySample.py`, `SCIU.py`, `decoder.py`: verbatim
  model code, with imports changed to package-relative (`from .SFE`,
  `from .AMCM`, `from .DySample`, `from .CSHF`).
- `SFE.py`: the official SFE architecture; only build-time mmcv/mmcls
  helpers were replaced (`build_norm_layer`, `ModuleList`,
  `resize_pos_embed`, `to_2tuple`, `BaseBackbone`, the `BACKBONES`
  registry decorator).  The model names `norm_final` instead of `norm1` to
  avoid a collision with SFE's `norm1` property during `add_module`.
- `patch_embed.py`: `ConvPatchEmbed` with the same constructor and
  `forward -> (tokens, (H,W))`; mmcv `AdaptivePadding(padding='corner')`
  is replaced by an explicit bottom/right pad.  For the 512 input the pad is
  zero at every stage.
- `cuda_new/`: the official `bi_wkv` C++/CUDA sources.  `SCIU.py` builds
  the extension at import; the hardcoded sm_86 flag is replaced by the
  current device capability or `SCRWKV_CUDA_ARCH`, with a clear error at
  forward if compilation fails.  The official target is RTX 3090 (sm_86).
- `__init__.py`: `build_scrwkv()` constructs the official `Crack`
  configuration (`embed_dims=256`, four SCIU layers, out indices 0–4) and
  the CSHF decoder.

## Invariants kept from the official code

- SFE stem/patch embedding, AMCM, SCIU blocks with GBST and DSCD dynamic
  decay, CSHF decoder, DySample upsampling, official 512-square input and
  fixed output scales (64/128/256/512).
- Official crack training: cv2 INTER_CUBIC resize of image and mask to
  512, mask threshold at 127, `(x/255 - 0.5)/0.5`, no augmentation,
  `0.25 * BCEWithLogits + 0.75 * Dice`, AdamW lr 5e-4, wd 0.01, PolyLR to
  1e-6, batch 1, 50 epochs.  The shared loader returns boolean 0/1 masks;
  the adapter multiplies by 255 before cubic resize so the threshold is
  equivalent to the official raw-mask path.
- Official inference (`test.py`): raw logits divided by the per-image
  maximum, threshold 0.2.
- The custom bidirectional WKV CUDA extension, not a substitution.

## Deviations / notes

1. mmcv/mmcls are build-time helpers only in the released SFE; they are
   replaced by local torch equivalents.  The architecture and forward path
   for the fixed 512 input are unchanged.
2. The CUDA extension is compiled on the target GPU (official sm_86).  The
   port selects the current capability or `SCRWKV_CUDA_ARCH` instead of
   hardcoding sm_86.  On a host without nvcc the model builds but forward
   fails with a descriptive error.  This is the only supported runtime path
   for WKV; no pure-PyTorch fallback was added.
3. The official 50 epochs are extended to the shared 800-epoch budget with
   validation-Dice selection; the shared runtime steps PolyLR per epoch,
   matching the released trainer.
4. Predictions are exported at the original GT resolution (the release
   evaluates 512 masks); the binary decision is made at 512 with the
   official rule before nearest resizing.
5. The port has ~2.04M parameters, whereas the paper reports 1.22M.  The
   released `arch_zoo['Crack']` configuration and CSHF/DySample modules are
   kept verbatim; the discrepancy is in the paper's count/config.
6. No official trained weights exist in the release; the model is trained
   from scratch on the shared 18-image split.  No network access is used.
