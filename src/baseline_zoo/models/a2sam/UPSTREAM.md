# A2SAM upstream record

## Sources

| Component | Upstream | Version |
|---|---|---|
| Paper | "Informativeness-driven active adaptation of SAM: Structural prompts and contrastive parameter selection for medical tubular segmentation", Medical Image Analysis 2026, DOI 10.1016/j.media.2026.104285 | 2026 |
| Code | `YongChen-Exact/A2SAM` (`3rd/A2SAM`) | default branch |
| SAM ViT-B | Meta `segment-anything` modules vendored inside A2SAM | ViT-B |

Licenses: A2SAM is MIT (`LICENSE`).  The SAM modules are Meta's
Apache-2.0 implementation (`LICENSE_SAM`).  The SAM ViT-B checkpoint is
loaded from the local `checkpoints/sam_vit_b_01ec64.pth`; the server is
offline.

## What is vendored

- `sam/{common,image_encoder,prompt_encoder,mask_decoder,transformer}.py`:
  the A2SAM repository's SAM ViT-B modules, copied verbatim with relative
  imports.  Only `tuning_mode='cps'` is used, which is the official A2SAM
  mode and does not construct the unused FFT/Adapter branches.

## Paper-level reimplementation (not in the release)

The released repository does **not** contain the paper's probabilistic U-Net
prior or the topological structural prompt generator.  It reads precomputed
`uncertainty`/`logit` `.npy` files, fuses them with a small Conv-BN-LeakyReLU
network, and detaches the fusion output from the loss.

This port implements the missing pieces:

- `prior.py`: a conditional probabilistic U-Net (Kohl et al., 2018 style)
  with a shared encoder, prior/posterior Gaussian heads, FiLM decoding and a
  KL term.  The prior is trained for 100 epochs on the 18 training images
  (Adam lr 1e-3, BCE + Dice + 1e-5 KL) and then frozen.  K=5 prior samples
  give the coarse likelihood (mean probability) and structural uncertainty
  (4 x variance, clipped to [0,1]).  GroupNorm is used instead of BatchNorm
  because the shared protocol trains with batch 1.
- `topology.py`: the likelihood map is skeletonized; 8-connected skeleton
  pixels are vertices, junction/endpoint pixels are critical cells, and
  degree-2 chains are branches.  A geodesic filtration from the most
  confident pixel supplies branch persistence; branches below an absolute
  (5 px) or relative (10%) threshold are pruned.  Disconnected components
  are processed independently.
- `spg.py`: retained branches are sampled every 3 pixels; a learnable
  branch gate scores each sample from `[x, y, probability, uncertainty,
  persistence, distance, length]`; samples are rasterized as Gaussian splats
  (sigma 2 px) at 256x256 and fused with the likelihood/uncertainty by the
  released Conv-BN-LeakyReLU pattern.  The prompt is **not** detached.
- `cps.py`: accumulated squared-gradient influence and magnitude
  preservation, global min-max normalization, score
  `alpha * influence/(magnitude+1) + (1-alpha) * influence`, top 3.5%
  elements kept, norms/position embeddings/IoU head frozen.  Statistics use
  the 18 training images rather than the released validation loader to avoid
  validation-label leakage.
- `loss.py`: weighted IoU and the released thickness-aware plus-kernel
  opening/EDT/skeleton term.  The divisor is the foreground pixel count
  (the release divides by all non-zero pixels after the sigmoid).

## Official recipe kept

- SAM ViT-B, 1024x1024 input, ImageNet normalization, horizontal flip only.
- AdamW lr 2e-4, weight decay 0.01, cosine to 1e-7 over the A2SAM stage,
  batch 1.
- BCE + IoU + 0.7 x thickness-aware loss.
- Dense automatic prompt (no manual points/boxes); SAM mask decoder with a
  256x256 soft mask prompt.
- CPS top 3.5% (alpha 0.4) applied once after the prior stage.
- Predictions are exported as 0/255 masks at the exact GT resolution.

## Protocol adaptations / assumptions

- 18 train / 6 validation images; 100 shared-budget epochs for the prior and
  700 for A2SAM (official 50 A2SAM epochs).
- Validation-Dice checkpoint selection (official selects the metrics of its
  test split).
- Prior input resolution 512 (the paper/release do not state it; the release
  precomputes maps at the network input size).
- The SPG's discrete-Morse construction is a graph-based approximation
  because the paper's exact DMC/DMT cell decomposition is not in the release
  and the paper full text is not available locally.
- Resume recomputes the CPS masks and prior/topology caches from the restored
  prior and SPG weights; the checkpoint stores the whole model (prior + SAM +
  SPG).
- There are no official A2SAM weights; the SAM ViT-B weights are the only
  pretrained asset.

## Verification

- CPU tests: prior shapes/KL/K-sampling, topology pruning, SPG prompt and
  gradients, CPS top-fraction/gradient masks, prior/IoU/thickness losses,
  256-input model forward, adapter batch/config round-trip.
- GPU smoke (1 prior + 1 A2SAM epoch, 2 train/1 val): SAM weights strict
  load (177/17/120 tensors), CPS kept 3.48% of 93.6M SAM elements, stage-2
  training and checkpointing completed, and prediction exported 16 DRIVE
  masks at 565x584 with 0/255 values.
