# KnowSAM

- Upstream: https://github.com/taozh2017/KnowSAM
- Vendored snapshot: `3rd/KnowSAM`, HEAD
  `b5de98d7f5538e39447d8f6e143c8292875cd22e`.
- License: no LICENSE file in the vendored snapshot; add the upstream
  license before redistributing this copy.
- Files copied and flattened:
  - `Model/{model,unet,vnet,discriminator,prompt}.py`
  - `Model/sam/`, `Model/common/`, `Model/ImageEncoder/`
  - `dataloader/{dataset,transforms,TwoStreamBatchSampler}.py` ->
    `dataset.py`, `transforms.py`, `sampler.py`
  - `utils/losses.py` -> `losses.py`, `utils/utils.py` -> `metrics.py`
  - `trainer.py`, `prediction_ACDC.py` -> `prediction_acdc.py`
- Modifications:
  - Package-relative imports; device-agnostic training/prediction (removed
    module-level `.cuda()` allocations and hard-coded CUDA device usage).
  - Added a guard for the consistency-ramp denominator and
    `np.ascontiguousarray` before tensor conversion in the official dataset.
  - The adapter seeds all RNGs with the protocol seed, stages labeled/val/
    unlabeled data from the split manifest, excludes the exact protocol
    source stems and every validation/test protocol group (CREMI volume,
    CHASEDB1 subject, CORN batch) from the unlabeled pool, and clears stale
    staging files.
- Extra dependencies: `medpy==0.4.0`, `hausdorff` (imported by the official
  metrics module).
- Runtime consumer: `baseline_zoo.adapters.knowsam.KnowSAMAdapter`
