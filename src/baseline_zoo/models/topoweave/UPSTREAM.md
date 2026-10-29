# TopoWeave

**Status: project extension, not an official port and not an official baseline.**

TopoWeave is a new single-stage method derived from the design lessons of
MaskVSC (TMI 2025). It does not copy MaskVSC source code and there are no
official TopoWeave or TopoWeave-compatible weights. No numerical parity with
MaskVSC is claimed.

## Provenance

- Inspiration and closest prior method: MaskVSC, *Masked Vascular Structure
  Segmentation and Completion in Retinal Images*, IEEE TMI 44(6):2492-2502,
  2025, DOI 10.1109/TMI.2025.3538336, code `Zhouyi-Zura/MaskVSC`.
- Local MaskVSC implementation: `src/baseline_zoo/models/maskvsc/`.
- New code: this package; no files copied from MaskVSC or `3rd/`.
- Reused repo-owned utilities: `core.losses.core.DiceLoss` and
  `SoftclDiceLoss` (verified soft-clDice/skeleton implementation), and
  `core.metrics` only for evaluation.
- Third-party dependencies already in the project: torch, numpy, scipy,
  scikit-image, Pillow. No new downloads or optional model weights.

## Why the upgrade

MaskVSC's audit and 3-shot runs exposed these problems:

1. `Connect_Loss` release mode is numerically inert (opening erases thin
   vessels; its denominator is effectively all pixels), while paper mode
   explodes on sparse domains.
2. TopoMask uses a fixed 7x7 dilation and global RNG and is tuned to retinal
   vessels.
3. The two-stage design is an information bottleneck: ViG sees only the
   stage-1 probability map, not image evidence.
4. The 23.8M ViG uses BatchNorm, takes ~100 epochs to stabilize, and improved
   validation Dice by only ~0.001 over stage 1 in the pooled 3-shot run.
5. Fixed lr and no scheduler are faithful to the release but poor for the
   800-epoch shared budget.

TopoWeave addresses each point:

| MaskVSC | TopoWeave |
|---|---|
| `Connect_Loss` | bounded BCE + Dice + soft-clDice, no opening/component count/per-foreground rescaling |
| fixed 7x7, global RNG | width-aware skeleton paths, explicit `random.Random`, biphasic ratio curriculum |
| frozen two-stage UNet + ViG | one model: GroupNorm ResUNet + jointly trained gated graph refiner |
| mask-only ViG input | graph tokens from image features plus a detached coarse probability map |
| 16 blocks, 23.8M, BatchNorm | 4 lightweight kNN blocks, LayerNorm, ~0.5M refiner params |
| fixed Adam lr | AdamW + cosine warm-up + gradient clipping |
| flip only | flips, rotation, scale, intensity/gamma/noise augmentation |

## Architecture and contract

- Input: RGB, 512x512, float [0,1].
- Backbone: GroupNorm residual U-Net, widths 32/64/128/256/512, four pools,
  bilinear decoder, 1x1 coarse head.
- Refiner (`--topoweave-refiner graph`, default): 1/16 image feature + detached
  coarse probability, 1x1 projection to 128 dims, fixed 2D sinusoidal position
  encoding, 4 kNN graph blocks (k=9, neighbor-minus-center max aggregation,
  LayerNorm, FFN). The correction is upsampled and fused with the full-res
  decoder feature and coarse probability; final logits are
  `coarse + scale * detail`, with `scale = 0.1 * sigmoid(gate)` (~0.05 at
  initialization). `--topoweave-refiner none` bypasses the refiner.
- Output: logits `[B,1,512,512]`; the adapter applies sigmoid and restores
  the original GT resolution with nearest interpolation.
- Parameters (512 input, default graph refiner): **8,389,284**; with
  `refiner=none`: **7,849,601**.
- Deep supervision: final logits get BCE + Dice + soft-clDice; coarse logits
  get BCE + Dice with weight 0.4.
- Optional `--topoweave-refiner-warmup N` disables the refiner for the first
  N epochs as a single-model training schedule; checkpoints are not eligible
  during that warm-up. This is not a frozen separate stage.

## Data protocol

- Training/validation: the shared `data/sacm_3shot_seed<seed>` 18/6 split.
- TopoMask uses training labels only; validation and test masks are never used
  for training or masking.
- 512x512 pooled 20-domain adaptation; predictions are compared at original
  GT resolution. FOV masking is not applied, matching the baseline-zoo
  evaluator used for every model.
- Validation-Dice selection; no test labels, no TTA.

## Verification

- `tests/test_topoweave.py`: registry/config, forward/backward and refiner
  switch, parameter range, TopoMask invariants and width awareness, bounded
  loss/empty targets, warm-up eligibility, deterministic augmentation, and a
  tiny CPU train + prediction through the shared runtime.
- `tests/test_baseline_zoo.py` expects the new 43rd config/available model.
- No official parity test is possible because no official TopoWeave code or
  weights exist; this is a project extension.

## Known limitations

- The graph refiner may be marginal or unstable on 18 training images; the
  `none` profile and refiner warm-up are first-class ablation options.
- Soft-clDice at 512 costs more than a pure BCE/Dice loss; the topology term
  is applied to the final output only.
- The method is evaluated outside MaskVSC's retinal scope (cracks, roads,
  materials); novel-domain results should be interpreted as domain shift, not
  as a MaskVSC reproduction.
- TopoWeave must not be counted as an official MaskVSC baseline or used to
  claim official MaskVSC parity.
