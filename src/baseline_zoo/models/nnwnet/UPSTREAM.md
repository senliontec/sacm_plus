# nnWNet

- Upstream: vendored `baselines/nnwnet` trainer plugin (URL not recorded in
  the README)
- License: no LICENSE file in the vendored snapshot; add the upstream
  license before redistributing this copy.
- Files copied:
  - `nnUNetTrainer_WNet2D.py` -> `trainer.py` (constructor adjusted to the
    nnU-Net 2.8.1 base signature; training logic/losses unchanged)
- External dependency: `nnunetv2==2.8.1`. The vendored `baselines/nnUNet`
  fork writes `.safetensors`, while the declared pip runtime writes `.pth`;
  the adapter discovers either extension and rejects safetensors at
  inference with the installed predictor. Training is seeded, staging and
  preprocessing are regenerated per workdir, and `--resume` is rejected
  because nnU-Net continue-training is not wired in.
- Runtime consumer: `baseline_zoo.adapters.nnwnet.NNWNetAdapter`
