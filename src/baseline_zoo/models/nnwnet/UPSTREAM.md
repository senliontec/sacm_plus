# nnWNet

- Upstream: https://github.com/Yanfeng-Zhou/nnWNet
- Vendored snapshot: `3rd/nnWNet`, HEAD
  `cfcd2e09754fc832c3099a2ed39d86a90007dbcb`; the trainer plugin is vendored
  into `baseline_zoo.models.nnwnet`.
- Paper: `parsing/[2025 cvpr] nnWNet: Rethinking the Use of Transformers in
  Biomedical Image Segmentation and Calling for a Unified Evaluation
  Benchmark` (Zhou et al., CVPR 2025).
- License: no LICENSE file in the vendored snapshot; add the upstream
  license before redistributing this copy.

## Paper vs project protocol

The WNet architecture matches the paper: two cascaded encoder-decoders,
OPE, 3x3 residual LSBs, pooling-based GSBs, channel concatenation, no
additional positional encoding. The 2D model has 7,044,042 parameters,
matching the paper's ~7M.

The paper's evaluation environment is deliberately not reproduced:

- paper: nnU-Net **2.3.1**, 200 epochs for 2D, 5-fold cross-validation,
  Jaccard/Dice/95HD/ASD;
- project: installed nnU-Net **2.8.1**, shared **800** epochs, fixed
  18/6 one-fold 3-shot split, shared Dice/IoU/clDice/HD95 evaluator.

nnU-Net 2.8.1 is kept because the stock `nnunet2d` baseline uses the same
environment; downgrading globally would change another baseline. The
vendored `nnUNetTrainer_WNet2D` constructor is adapted to the 2.8.1 base
signature. The paper's 5-fold/ASD protocol is not implemented because the
benchmark has one fixed 18/6 split and a shared metric set. These are
documented deviations, not claims of paper-table reproduction.

## Monitoring and checkpoint selection

The training W&B run now uses the shared run_config fields and the same
`val/*` schema as other baselines at full-validation epochs:

- every `--nnwnet-val-interval` epochs (default **1**, fully per-epoch
  curves), the adapter runs nnU-Net full-image validation (sliding window +
  TTA) on the 6 val cases;
- predictions are evaluated with the shared `binary_metrics`
  (Dice/IoU/clDice/HD95 and per-dataset keys);
- checkpoint selection uses the shared full-image validation Dice and writes
  `checkpoint_benchmark_best.pth`; prediction prefers that checkpoint, then
  falls back to nnU-Net `checkpoint_best`/`checkpoint_final`;
- nnU-Net's patch-based EMA pseudo-Dice remains logged only under
  `nnunet/*`; it is not used for selection;
- off-interval epochs log `train/loss`, `lr` and `nnunet/*`; the final epoch
  is always fully validated.

Full validation adds nnU-Net inference time every N epochs. The default
`--nnwnet-val-interval 1` matches the per-epoch schema of other baselines;
set a larger value to reduce cost (off-interval rows then show train/loss
only). The final epoch is always validated.

- Files copied:
  - `nnUNetTrainer_WNet2D.py` -> `trainer.py` (constructor adjusted to the
    nnU-Net 2.8.1 base signature; training logic/losses unchanged)
- External dependencies: `nnunetv2==2.8.1` and `timm` (the WNet trainer
  imports timm layers directly). The reference nnU-Net fork writes
  `.safetensors`, while the declared pip runtime writes `.pth`; the adapter
  discovers either extension and rejects safetensors at inference with the
  installed predictor.
- Workdir reuse: raw/preprocessed data, this dataset's `results` tree,
  `nnunet_model_dir.txt`, and per-dataset prediction input/output/export
  directories are cleared before the corresponding stage. This prevents a
  stale `checkpoint_best` or stale prediction PNG from being picked up.
  Training is seeded; `--resume` is rejected because nnU-Net
  continue-training is not wired in.
- Channel policy: one nnU-Net dataset needs a uniform channel count. The
  default `--nnwnet-channels rgb` converts every case to RGB (grayscale is
  replicated); `grayscale` converts every case to one channel for
  all-grayscale protocols. The selected policy is recorded in the workdir
  and model folder and reused at prediction.
- Runtime consumer: `baseline_zoo.adapters.nnwnet.NNWNetAdapter`
