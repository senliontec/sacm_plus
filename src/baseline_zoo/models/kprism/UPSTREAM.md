# K-Prism upstream record

- Upstream: https://github.com/bangwayne/K-Prism
- Local reference tree: `3rd/K-Prism` (never imported at runtime)
- Commit: `78c3b1bc0fbfb12434dde01a3f888b92b455293c`
  ("Update README.md", 2026)
- License: MIT (upstream `LICENSE`, Copyright 2025 bangwayne; copied to
  `src/baseline_zoo/models/kprism/LICENSE`). `point_encoder/common.py` is a
  small SAM/detectron2-derived MLP/LayerNorm utility as distributed in the
  upstream MIT tree.
- Paper: ICLR 2026. No paper PDF/full text was present in the reference
  tree; the README and released code were the implementation source.
- Released weights: none. The port trains from scratch.
- Dependencies in the copied semantic path: torch, numpy, cv2, einops,
  albumentations 1.3.1. detectron2/fvcore/omegaconf are not installed or
  used.

## Files copied

```
backbone/utnet.py
backbone/conv_trans_utils.py
backbone/unet_utils.py
pixel_fuser/pixelfuser.py
transformer_decoder/transformer_decoder.py
transformer_decoder/SimpleMaskDecoder.py
transformer_decoder/position_encoding.py
meta_arch/seghead.py
point_encoder/common.py
point_encoder/point_encoder.py
point_encoder/point_feature_map_encoder.py
loss.py
```

`modeling/task_encoder/` (`MaskEncoder`, `ObjectSummarizer`) and
`modeling/criterion.py` were not copied: they exist only for in-context
(`click_mode` 3/4) and the task-encoder branch. The benchmark port is the
prompt-free semantic branch (mode 0).

## Modifications to copied files

- Replaced `detectron2.modeling.ShapeSpec`, `detectron2.layers.Conv2d`,
  `detectron2.layers.weight_init.c2_xavier_fill` and `fvcore` imports with
  local equivalents in `compat.py` (all call sites use the plain forms).
- Removed `omegaconf.DictConfig`; constructors take the local `CfgNode`
  (a dict with attribute access) instead.
- Removed debug-only `pdb` and duplicate `matplotlib` imports, and restored
  missing `typing`/`Tensor` imports that the upstream wildcard imports had
  supplied.
- No model math, layer definitions, tensor shapes or loss formulas were
  changed.

## Added files

- `model.py`: semantic assembly (`UTNet` + `SegHead` + `PointSampleLoss`),
  binary config, zero dense-prompt forward. `num_classes=2`, foreground
  query label 1, 6 decoder layers, 5 MoE experts, 512 input.
- `transforms.py`: upstream albumentations 1.3.1 training pipeline and
  official eval resize+centre-pad+normalize (mean 0.26, std 0.30).
- `compat.py`: small detectron2/fvcore utility replacements.
- `__init__.py` files for the package tree.
- `src/baseline_zoo/adapters/kprism.py`: runtime adapter.

## Known upstream issues not used by this port

- `KPrism.semantic_inference` is a broken staticmethod (takes `self`); the
  adapter uses `pred_masks` directly.
- Official semantic mode still derives initial points from ground-truth
  masks via `PointSampler.initial_test_points`; this port passes a zero
  dense map instead.
- `MultiScaleMaskedTransformerDecoder` supports click and ICL modes that
  require `MaskEncoder`/`ObjectSummarizer`; those branches are absent here
  and are never called.
- The official scheduler sets the first epoch LR to zero; this port starts
  at 1/10 of the base LR (documented in the adapter deviations).
- `PointSampleLoss` computes an unused `num_masks = B*16`; its Dice/CE
  helpers normalize per batch, so binary semantic training is unaffected.
- UTNet also has a 256-input branch, but its level channels (48/96/192) do
  not match the released 96/192/384 config. The port rejects non-512 input.

## Input/output contract

- Input: RGB `[B,3,512,512]`, normalized `(x/255 - 0.26)/0.30`.
- Dense prompt: zeros `[B,1,3,H,W]` (upstream uses GT-derived point maps).
- Output: `pred_masks [B,1,128,128]` plus five deep-supervision aux outputs.
- Prediction: logits upsampled to 512, sigmoid, remove centre padding,
  bilinear resize to the original GT size, threshold 0.5.
