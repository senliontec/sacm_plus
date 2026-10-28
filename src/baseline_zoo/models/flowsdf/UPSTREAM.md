# FlowSDF upstream record

## Sources

| Component | Upstream | Version |
|---|---|---|
| Paper | "FlowSDF: Flow Matching for Medical Image Segmentation Using Distance Transforms", IJCV 2025 | 2025 |
| Code | `leabogensperger/FlowSDF` (`3rd/FlowSDF`) | default branch |
| SDF preprocessing reference | `leabogensperger/generative-segmentation-sdf`, `preprocess_data/precompute_sdf.ipynb` | not vendored |

License: MIT (see `LICENSE` in this directory).

## What is vendored

- `unet.py`: `UNetModel` with timestep embedding, residual blocks, attention
  blocks, up/down sampling and the RRDB image encoder injection.  The unused
  FP16 conversion methods and `SuperResModel` are omitted.
- `rrdb.py`: RRDB / ResidualDenseBlock5C image encoder.  Upstream defaults
  `in_nc=3`; the constructor is parameterized and the adapter passes 3.
- `nn.py`: SiLU, GroupNorm32, conv/linear/pool helpers, sinusoidal timestep
  embedding, activation checkpointing.

## Official training recipe

- Input: 128x128 random crops, RGB image scaled to `[-1, 1]`, random
  horizontal and vertical flips.  MoNuSeg images are colour-normalized and
  resized to 500x500 before cropping; GlaS images are resized to 128.
- Target: truncated signed distance map, negative inside, positive outside,
  truncated at delta=15 and normalized by delta.  The official repository
  reads precomputed `.npy` SDFs from the companion notebook; this port
  computes the exact Euclidean truncated SDF on the fly.
- Objective: conditional flow matching, `t ~ U[0,1]`,

  ```
  sigma_t = 1 - (1 - sigma_min) t
  m_t     = t * m + sigma_t * eta,          eta ~ N(0, I)
  u       = (m - (1 - sigma_min) * m_t) / (1 - (1 - sigma_min) * t)
  loss    = mean((v_theta(m_t, t | x) - u)^2)
  ```

  with `sigma_min = 1e-5`.
- Optimizer: Adam, lr 1e-4, no scheduler, gradient clip 1.0 (inf-norm).
- Batch size: 4 in the released config; the paper states 16.  This port keeps
  the released batch 4 on the 18-image split.
- EMA of all parameters, decay 0.999 in the released config (paper 0.9999).
- Checkpoint selection upstream: lowest training loss.  This port selects by
  validation Dice of the EMA weights (shared protocol).
- Sampling: Euler integration of the probability-flow ODE from Gaussian noise
  at `t=0` to the SDF at `t=1`, `nfe=4`.  Threshold `sdf <= 3e-2`.

## Protocol adaptations

- 18 train / 6 validation images, shared 800-epoch budget, validation-Dice
  selection.
- Training: random 128 crops (resize up if the short side is smaller than
  128), random H/V flips.  No Vahadane colour normalization (not part of the
  released code).
- Final prediction: overlapping 128x128 tiles with stride 64, averaged in
  SDF space (the MoNuSeg patch-averaging strategy in the paper), then resized
  to the exact GT resolution and thresholded.  `K=4` independent samples are
  averaged per tile (MMSE).
- Per-epoch validation: the whole image is resized to 128 and sampled once
  (the official GlaS path), then the SDF is resized back to GT for scoring.
  This keeps the shared every-epoch validation affordable; the one-time test
  prediction still uses the tiled MoNuSeg path.
- EMA weights are used for validation and prediction; checkpoints store both
  raw and EMA states so training can resume.
- The empty `SimulationHelper` submodule, TensorBoard and `torchdiffeq` are
  not used; the shared runtime owns checkpointing/logging and a local Euler
  loop replaces `odeint`.
- The released `sampler.py` passes `torch.linspace(0, 1, 4)` (three Euler
  steps) although the config labels it "4 NFE"; this port runs four equal
  Euler steps as specified in Algorithm 2.
- Upstream reads images with OpenCV (BGR); this port uses RGB.  Training is
  from scratch, so channel ordering does not change the model class.
- Upstream evaluation treats a four-sample batch as four separate
  predictions; this port averages samples per image (MMSE) as described in
  the paper.

## Assets / verification

- No pretrained weights; trained from scratch.
- Local smoke: CPU SDF/sampler tests, a one-epoch GPU run and a one-image
  prediction at exact GT resolution.
