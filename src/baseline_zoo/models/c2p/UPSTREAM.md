# C2P upstream record

## Sources

| Component | Upstream | Commit |
|---|---|---|
| Network / geometry-aware inference | `Yundi218/Concept-to-Pixel`, `network.py`, `train.py` (`predict_with_token_tta_fast`) | `5d33f8e514ea50013619d8d0011deb97b7453c8e` |

Paper: "Concept-to-Pixel: Prompt-Free Universal Medical Image Segmentation",
ECCV 2026, arXiv:2603.17746.

License: the checkout has no LICENSE file, although the README badge says
MIT.  Verify redistribution terms with the authors before releasing.

## Copied and adapted

- `network.py`: verbatim C2P architecture (`PositionalEncoding2D`,
  `FeedForward`, `StyleContentModalityLearner`, `CrossAttention`,
  `BidirectionalCrossAttentionLayer`, `TokenImageTransformer`, `C2P`).
  No code changes; the timm backbone is created by the adapter with
  `pretrained=False`.
- `__init__.py`: `build_c2p` loads the official checkpoint from disk,
  unwraps optional `model`/`state_dict`/`model_state_dict`/`generator`
  containers, strips `module.` prefixes, ignores the checkpoint's
  training-only `ds_head4.*`, `ds_head5.*`, `d2_proj_inv.*` parameters, and
  fails on any other missing/unexpected key.  `random_init=True` is a
  smoke-only path.

## What is kept from the paper / official code

- ConvNeXtV2-B features, shallow/deep style-content modality learner,
  9 geometric tokens and 9 semantic tokens, 3-layer bidirectional
  token-image transformer (d=584), token-guided dynamic head with
  foreground/background kernels, and the 9 geometric prediction heads.
- The official input pipeline: 384-square resize, ImageNet normalization.
- The official geometry-aware inference consensus from
  `predict_with_token_tta_fast`: identity/hflip/vflip views, per-view
  sigmoid probability, weighting by `exp(-5 * (10 * |mask_area -
  token_area| + ||mask_centroid - token_centroid||))`, inverse-view
  unflipping, weighted average, and empty-prediction suppression when the
  mean token area < 0.001 and the mean probability < 0.005.
- Prompt-free inference: no support images, points, boxes or target GT.

## Why the port is zero-shot

The released repository is not runnable end-to-end: `train.py` imports a
non-existent `SRICL` class, and the README's `train_sem.py`/`evaluate_sem.py`
are absent.  The full network and the official HuggingFace checkpoint
(`yunyundi/c2p`, file `c2p.pt`) are available.  The paper's training
additionally needs precomputed geometric property caches and
MLLM/PubMedBERT semantic embeddings for the training datasets; those
generators and the 8 medical datasets are not part of this checkout.
Re-training C2P on the shared 18-image split without the semantic
distillation targets would not be a faithful C2P.  The port therefore
evaluates the official checkpoint zero-shot, matching the paper's
universal/zero-shot usage, and discloses that the shared 18-image split is
not used.

The official checkpoint is 1.69 GB with sha256
`5f4a22aa892ea1069568fd846a1b726a01fd2b8f45020a0d54a63f32c62093a0`;
it is placed at `checkpoints/c2p/1.pth`.  Its wrapper is
`{epoch, model_state_dict, optimizer_state_dict}`; the loader unwraps
`model_state_dict` and ignores the training-only auxiliary keys listed
above.

## Deviations / notes

1. Zero-shot evaluation of the official 8-medical-dataset checkpoint; no
   fine-tuning on the shared split, no support images, no target labels.
2. The checkpoint supplies the ConvNeXtV2-B backbone; timm is constructed
   with `pretrained=False`, so the offline server never downloads weights.
3. Inference is fp32 (the official trainer uses AMP).  TTA is on by default
   (`--c2p-tta 1`); disabling it gives the official single-view path.
4. The output is a sigmoid probability resized to the original GT
   resolution and thresholded at 0.5.
5. The official C2P checkpoint is required at `checkpoints/c2p/1.pth`;
   `--c2p-random-init` is smoke-only.  The loader accepts the released
   `model_state_dict` wrapper and ignores training-only auxiliary keys.
6. Applying a medical-trained model to the protocol's natural-image
   curvilinear domains is a deliberate zero-shot generalization test, not a
   claim of in-domain performance.
