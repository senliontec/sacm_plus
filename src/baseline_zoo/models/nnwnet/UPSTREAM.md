# nnWNet

- Upstream: vendored `baselines/nnwnet` trainer plugin (URL not recorded in
  the README)
- License: no LICENSE file in the vendored snapshot; add the upstream
  license before redistributing this copy.
- Files copied:
  - `nnUNetTrainer_WNet2D.py` -> `trainer.py` (unchanged)
- External dependency: `nnunetv2==2.8.1`. The copied trainer's constructor
  was adjusted to the 2.8.1 signature (the upstream `unpack_dataset`
  argument was removed); training logic, architecture and losses are
  unchanged. The package is installed via `pip`; it is not part of the
  disposable `baselines/` tree.
- Runtime consumer: `baseline_zoo.adapters.nnwnet.NNWNetAdapter`
