# TPSegformer upstream record

## Sources

| Component | Upstream | Version |
|---|---|---|
| Paper | "Hilbert Curve-Based Attention Enabling Topology-Preserving Image Tensor Representation for Semantic Segmentation Network", CVPR 2026 | 2026 |
| Code | `mumu-k/TPSegformer` (`3rd/TPSegformer`) | default branch |
| Paper parsing | `parsing/[2026 cvpr] Hilbert Curve-Based Attention ...` | local |

License: the upstream repository has **no LICENSE file**.  The vendored
decode-head modules are adapted from that repository; verify redistribution
rights before publishing.

## What is vendored

- `modules.py`: SEM (`eca_layer`: ECA channel attention + 3x3 BN/ReLU/dropout
  spatial branch), HLFF (depthwise-separable 3x3 on the concatenated
  X3/X4-upsampled features, 1152 -> 384), and LCCM (3x3 384 -> 5,
  ReLU+BN, channel max, 1x1 QKV, curve flattening, Q^T K softmax, V
  weighting, inverse curve, sigmoid gate).
- `hilbert.py`: the release's Hilbert traversal without the external
  `hilbertcurve` package.  The iterative `xy2d` index is cached per square
  size; row-first and Z-order traversals are kept for ablations.
- `head.py`: TPHead and the five-conv FCN auxiliary head.
- `loss.py`: squared-denominator Dice (eps 1e-3) and Sobel edge-aware L1.

## Defects fixed in the release

- `TPHead.forward` calls `self.cls_seg`, which does not exist
  (`BaseDecodeHead` defines `conv_seg`).  The port defines an explicit
  1x1 classifier from the five class-specific channels to the binary output.
- The auxiliary FCN head is configured with 19 classes while BD3 has five
  and the paper head has five; the port uses one binary channel.
- `lccm.py` imports `hilbertcurve`, `thop` and the absolute module
  `hbtrans`; all three are replaced by the local Hilbert implementation.
- The upstream config points Swin-T at an OpenMMLab URL and MMSeg's
  `pretrained=` path.  The port never downloads; timm Swin-T is built with
  `pretrained=False`.

## Official recipe kept

- Swin-T (depths 2/2/6/2, heads 3/6/12/24, window 7, patch 4, drop path
  0.3), 512-square input, mean `[0.511, 0.539, 0.556]`, std
  `[0.084, 0.082, 0.083]`, BGR->RGB (the loader is RGB).
- TPDecoder: SEM on X3/X4, HLFF, LCCM with five class-specific layers and
  Hilbert traversal, 1x1 classifier.
- Five-conv FCN auxiliary head (BN, dropout 0.1) on X3, Dice 0.5 +
  edge-aware 1.0; main cross-entropy.
- AdamW lr 6e-5, betas (0.9, 0.999), wd 0.01, head lr x10, norm/position
  parameters with zero weight decay, batch 4.
- Horizontal flip and PhotoMetricDistortion augmentation.

## Protocol adaptations / assumptions

- Main CE is replaced by binary BCE; the auxiliary losses are unchanged.
- Official RandomResize/RandomCrop/pad and PhotoMetricDistortion are
  approximated by a 512-square bilinear resize (nearest GT), hflip and a
  light brightness/contrast/saturation/hue jitter.
- Official per-iteration LinearLR(1500) + PolyLR(160k) becomes a per-epoch
  5-epoch warmup starting at 1e-6 of the base LR, then linear poly decay
  over the shared 800-epoch budget (the runtime steps the scheduler once
  per epoch).
- No TTA or Dacl10k 512/341 sliding-window test; predictions are bilinearly
  upsampled to the exact GT resolution and thresholded.
- The official OpenMMLab Swin-T checkpoint is auto-detected under
  `checkpoints/*swin*tiny*` and its MMSeg `stages`/`attn.w_msa`/`ffn.layers`
  keys are mapped to timm; training falls back to random initialization
  only if no checkpoint is present or `--tpsegformer-backbone-ckpt` is
  explicitly omitted.
- The paper targets UAV building defects; this is a task adaptation to
  curvilinear structures.

## Verification

- CPU tests: Hilbert permutation/inverse and locality, LCCM Hilbert vs
  row-first equivalence with shared weights, head/model forward, losses,
  adapter data/config, optimizer group LRs/decays, prediction shape.
- GPU smoke: 4 train / 1 val images at 512, one epoch in 2 s, 34.6M params,
  checkpoint saved.
- Prediction smoke: 16 DRIVE masks exported in 2 s at 565x584 with 0/255
  values.
