# SACM (official)

## Source

- Repository: `kylechuuuuu/SACM`
- URL: https://github.com/kylechuuuuu/SACM
- Local reference snapshot: `3rd/SACM` (read-only third-party tree)
- Snapshot commit: `986504957e3f41f7fc02e610e2ad9ad27c43300f`
  (`Change device to use CUDA if available`, 2026-07-07)
- License: the upstream snapshot has no `LICENSE` file. Do not redistribute
  the vendored copy without obtaining a license from the authors.
- Vendored path: `src/baseline_zoo/models/sacm_official/segment_anything/`

## Copied files

- The complete upstream `segment_anything/` package, including
  `build_sam.py`, `predictor.py`, `automatic_mask_generator.py`,
  `modeling/` and `utils/`.

## Modifications

- Imports are package-relative. The upstream `predictor.py` absolute
  `segment_anything.modeling` import is rewritten.
- `modeling/image_encoder.py` gains an optional per-block
  activation-checkpointing branch used by the adapter; it changes activation
  memory, not the computation.
- No model parameter names, tensor shapes or computation were changed.

## Official recipe

- `train_sam.py`: AdamW 3e-4 with weight decay 0.01, cosine annealing to
  1e-6, 60 epochs, gradient clipping 1.0, BCE 0.7 + Dice 0.3, frozen image
  encoder except adapters, zero-prompt decoder forward.
- `test_sam.py`: prompt-free decoder, sigmoid-free threshold at 0.5 on raw
  logits, then the prediction is resized to the original image size
  (`test_sam.py:260-261`) before scoring. Its `SegmentationDataset(root_dir)`
  reads `images/` and `masks/` and has no split argument; the port evaluates
  held-out 20-dataset test roots.
- The port preserves the optimizer, schedule, loss weights, freeze policy
  and zero-prompt forward. It applies sigmoid before thresholding (the
  official raw-logit threshold is an inference bug) and selects the
  checkpoint by validation Dice rather than the upstream validation F1/loss.
- The shared evaluator consumes GT-resolution masks and reports
  Dice/IoU/clDice/HD95; the official script reports F1/precision/recall.

## Runtime consumer

`baseline_zoo.adapters.sacm_official.SACMOfficialAdapter`
