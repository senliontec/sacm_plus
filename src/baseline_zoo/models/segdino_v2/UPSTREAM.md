# SegDINO v2 upstream record

## Sources

| Component | Upstream | Commit / version |
|---|---|---|
| Official v2 decoder | `script-Yang/segdino_v2`, `dpt.py` | `0965c79` (local clone) |
| DINOv3-S/16 backbone | timm `vit_small_patch16_dinov3`, shared with the SegDINO v1 port | timm 1.0.22 |
| DINOv3 weights | local `checkpoints/dinov3/pytorch_model.bin` | ViT-S/16 |

Paper: "SegDINO: Introducing Multi-Scale Structure into DINO for Efficient
Medical Image Segmentation", arXiv 2606.17972 / MICCAI 2026 early acceptance.
The paper folder is not present under `parsing/`; the port was checked against
the archival paper and the official repository.

License: Apache-2.0 (`LICENSE`, copied from the upstream repo).

## What was copied / reconstructed

- `dpt.py`: the official `ResidualDepthwiseBlock`, `TPAResampleProject`,
  `TPASADDecoder` and `DPT` classes, with one fallback: the local timm
  DINOv3 backbone exposes intermediate features through
  `forward_intermediates` rather than the official local-hub
  `get_intermediate_layers` API.
- `__init__.py`: a self-contained builder that creates the timm DINOv3-S/16
  backbone, loads the local weights and wraps the official decoder.
- Not copied: the official `train.py`/`test.py`/`mydataset.py`/config files;
  the adapter provides the shared-protocol training/eval logic.
- The backbone loader (`find_dinov3_checkpoint`) is the public utility already
  used by the SegDINO v1 port; the actual network is timm.

## Invariants kept

- TPA: four 1x1 token projections to 256 channels, bilinear branches at
  scale 8/4/2/1, then 3x3 convs.
- SAD: per-branch residual depthwise block (depthwise 3x3, pointwise 1x1,
  GroupNorm(32), GELU, gamma=0) and top-down bilinear fusion, 1x1 output.
- Input 256 square, ImageNet mean/std.
- Loss BCEWithLogits; AdamW lr 1e-4 (paper weight decay 1e-4).
- Official augmentation (H/V flips, ±15° rotation, ColorJitter with
  brightness/contrast 0.2, saturation 0.1, hue 0.05) when enabled.
- Inference threshold 0.5; no TTA.

## Deviations

1. The paper says a frozen DINOv3 encoder, CE and no augmentation; the
   released trainer trains all parameters, uses BCE and applies the
   augmentation above. The port defaults to **frozen encoder** and released
   BCE/augmentation; flags `--segdino-v2-train-backbone` and
   `--segdino-v2-no-augment` reproduce the other choices.
2. Paper weight decay 1e-4 vs released AdamW default 0.01; the port uses
   1e-4.
3. Official 50 epochs -> shared 800 epochs with validation-Dice selection;
   the official trainer also selects best validation Dice.
4. The official loader uses a local `torch.hub` DINOv3 checkout; the port
   uses the same timm DINOv3-S/16 backbone as SegDINO v1.
5. Official metrics are computed at 256; the port upsamples logits to the
   exact GT resolution before thresholding.
6. The paper describes TPA resolutions 1/4..1/32 and 27.68M/6.08M parameters;
   the released code uses 1/2..1/16 and ~24.9M/3.3M with decoder_dim 256.
   The released code is kept.

## Assets

- Local DINOv3-S/16 weights at `checkpoints/dinov3/pytorch_model.bin`
  (or an explicitly named size-matched file).
- No official SegDINO v2 trained weights exist; training starts from the
  DINOv3 backbone. `--segdino-v2-random-init` is smoke-only.

## Verification

- `tests/test_segdino_v2.py` covers registry/metadata, random build and
  forward, freeze policy, preprocessing, loss, config round-trip and the slow
  real DINOv3 forward.
- One-epoch GPU smoke and a one-image prediction (0/255 PNG at GT size) were
  run locally.
