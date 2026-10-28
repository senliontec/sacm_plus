# nnU-Net 2D upstream record

## Sources

| Component | Upstream | Version |
|---|---|---|
| nnU-Net planner, preprocessor, 2D trainer, predictor | `MIC-DKFZ/nnUNet` (`nnunetv2`) | installed in the project env |
| Paper | "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Nature Methods 18:203-211 (2021) | 2021 |
| Network | `dynamic_network_architectures` `PlainConvUNet` | dependency of nnunetv2 |

License: nnU-Net is Apache-2.0. No upstream source is vendored here; the
adapter calls the installed package's official APIs.

## What is used

- Official 2D experiment planner and `DefaultPreprocessor` (ZScore
  normalization, 0.5 foreground oversampling, patch size auto-planned).
- Stock `nnUNetTrainer`: CE + Dice, deep supervision, SGD Nesterov 1e-2,
  weight decay 3e-5, PolyLR 0.9, AMP, gradient clip 12, 1000 epochs,
  EMA pseudo-Dice checkpoint selection.
- Official `nnUNetPredictor`: 0.5-overlap sliding window, Gaussian
  weighting, mirroring TTA, resampling to the original geometry.

## Protocol staging (the only custom code)

- `Dataset502_Curvilinear3Shot` with the shared 24 cases (18 train + 6 val),
  RGB PNG images and 0/1 labels, natural-image reader.
- Plans and normalization are fitted on the 18 train cases only (`--no_pp`);
  the 6 val cases are staged after planning and preprocessed with the frozen
  plans. `splits_final.json` has one fold: 18 train / 6 val.
- The official 1000-epoch default is replaced by the shared 800-epoch budget;
  the trainer still reports its patch-based pseudo Dice and selects the
  checkpoint by EMA pseudo-Dice. The final evaluator computes Dice/IoU/clDice/
  HD95 on exported masks.
- Predictions are 0/1 label maps, exported as 0/255 PNGs at the exact GT
  resolution.
- `--resume`, `--batch-size` (planner-controlled), `--accum-steps`,
  `--grad-clip`, `--selection` and `--num-workers` are rejected by the custom
  pipeline because they are owned by nnU-Net.

## Assets / verification

- No pretrained network or checkpoint; training is from scratch. The model is
  the stock 2D `PlainConvUNet` planned by nnU-Net (the local smoke planned an
  8-stage, 32->512, 640x768-patch network with batch 2).
- One-epoch GPU smoke (18 train / 6 val) and a one-image prediction (0/255
  PNG at 565x584 GT size) were run locally.
