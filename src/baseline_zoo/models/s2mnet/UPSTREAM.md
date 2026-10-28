# S2M-Net upstream record

## Sources

| Component | Upstream | Version |
|---|---|---|
| Paper | "S2M-Net: Spectral-Spatial Mixing with Morphology-Aware Adaptive Loss for Medical Image Segmentation", ICML 2026 | 2026 |
| Code | `sanaullah-ashfat/S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation` (local `3rd/...`, TensorFlow/Keras) | default branch |
| Paper parsing | `parsing/[2026 icml] S2M-Net: ...` | local |

License: the upstream repository has **no LICENSE file**.  The PyTorch port
is a translation of the released architecture/loss; verify redistribution
rights before publishing.

## Translation map

| Release (Keras) | Port (PyTorch) |
|---|---|
| `SpectralSelectiveTokenMixer` (`blocks.py`) | `SpectralSelectiveTokenMixer` (`blocks.py`) |
| `MRF_SE_Block` | `MRFSEBlock` |
| `BFP_DecoderStage` | `BFPDecoderStage` |
| `S2MNet` model function | `S2MNet` class (`model.py`) |
| `MorphologyAwareLoss` + five components | `MorphologyAwareLoss` and components (`loss.py`) |

The math is unchanged: 16-channel stem; five strided stages
(24/32/64/80/128); MRF-SE 1x1 expansion x6, parallel depthwise 3/5/7,
concat/fuse, SE-16, 1x1 projection, residual; SSTM per-channel FFT with a
learnable KxK real filter, resize to/from K=min(32,H,W), gated dense
projection on stages 3-5, concat/1x1/LayerNorm fusion and residual; BFP
region convs, boundary map, boundary-attended refinement and soft routing;
2x nearest head, 32/16/1 convs and sigmoid.

Keras defaults that matter are matched: Conv2D biases, ELU with alpha 1,
BatchNorm eps 1e-3 / momentum 0.01, elementwise dropout, LayerNorm over the
channel axis, and sigmoid probabilities.

## MAL

The five components are reproduced from the release:

- core: 0.4 Dice + 0.3 IoU + 0.3 boundary-weighted BCE (weights 1+5b);
- boundary: multi-scale gradient L1 with scale weights 0.5/0.3/0.2;
- structure: compactness difference;
- scale: focal loss with gamma 3/2/1.5 from object-size thresholds;
- texture: Laplacian difference.

The learnable scalar weights start at core=1, boundary=1, structure=1,
scale=0.5, texture=0.5 and are clipped to [0.1, 10].  The adapter clamps them
after every optimizer step.  The morphology modulation factors and
coefficients are unchanged.  TensorFlow's L2 kernel regularizer (1e-4) is
implemented as Adam weight decay 1e-4 on all parameters, which is a broader
but practical PyTorch equivalent of the Keras per-kernel regularizer.

## Protocol adaptations / assumptions

- The official retinal pipeline (CLAHE, green-channel conversion, FOV mask,
  256-pixel patches) is not applied to the mixed protocol: RGB [0,1],
  352-square random crops, flips/rot90 and light brightness/contrast
  jitter.
- Inference uses overlapping 352 patches (stride 176) with probability
  averaging; the release's retinal stride is 32.  Predictions are resized
  to the exact GT resolution.
- Official Adam 1e-4, clip 1.0, cosine with 10 warmup epochs and 150 epochs
  become the shared 800-epoch budget with per-epoch warmup+cosine.  The
  official effective batch 32 becomes batch 8 on the 18-image split.
- Binary sigmoid head only.  No official weights; random initialization.
- The release's "SSM" path is a gated dense projection, not a selective
  scan; the port preserves that released behavior.

## Verification

- CPU tests: model/block forward shapes and 4.77M parameter count, MAL
  finiteness/gradients and weight clipping, adapter batch/config,
  optimizer/scheduler, tiled prediction shape (including a resized small
  image).
- GPU smoke: 4 train / 1 val at 352, one epoch in 1 s, checkpoint saved.
- Prediction smoke: 16 DRIVE masks exported in 2 s at 565x584 with 0/255
  values.
