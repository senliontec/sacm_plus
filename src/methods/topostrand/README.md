# TopoStrand-SAM

Independent 3-shot curvilinear segmentation model built on a physically copied
frozen SAM ViT-L + four-candidate decoder scaffold. It keeps the 20-dataset
3-shot protocol and fixed candidate-3 deployment, and adds two model-level
changes aimed at the failure modes of DCD/SPA-SAM/CurviMix:

## 1. Width-calibrated deployment (WCD)

The deployment slot (slot 3) receives two additional training terms:

- precision-heavy Tversky, `alpha=0.7, beta=0.3`, weight 0.1;
- foreground-area calibration, L1 between predicted and target foreground
  fraction, weight 0.1.

The DCD/CurviMix role loss on slot 3 is recall-heavy (`alpha=0.3,beta=0.7`).
Under 3-shot domain shift the predicted foreground area is often 2.8-6.3x the
GT area on Novel thin structures. The two terms above calibrate
precision/foreground width without changing the architecture or inference
contract. On the seed-0 50-epoch pilot this improved all four macros versus
CurviMix-50:

| model | Dice | IoU | clDice | HD95 |
|---|---:|---:|---:|---:|
| CurviMix-50 noTTA | 59.12 | 45.82 | 67.53 | 62.12 |
| TopoStrand-50 noTTA | **59.58** | **46.09** | **67.84** | **59.35** |

CAPE path loss was tested at 50 epochs and was negative
(58.55 / 45.09 / 67.89 / 61.38); it is an ablation, not the main recipe.
Copy-paste scale jitter is also optional and off in the canonical recipe.

## 2. Resolution-consensus decoding (RCD)

For images larger than the SAM input, evaluate a full-image view and
native-resolution overlapping tiles. The output is selected without target
labels from prediction statistics:

- tile/full skeleton ratio `<= 1.0` -> full view;
- tile/full area ratio `< 0.7` -> tile view;
- otherwise -> full ∩ tile agreement.

This recovers thin native-resolution structures on high-resolution images
while avoiding the catastrophic tiling failure on coarse-GT FIVES. On the
seed-0 CurviMix checkpoint, this routing improved the 20-dataset macro from
60.50/47.01/69.24/58.97 to about 61.4/48.0/69.5/58.0 without using test
labels. The exact routed tables are produced by `evaluate_routed.py`.

## Paths

- Adapter/model: `src/methods/topostrand/adapter.py`,
  `src/methods/topostrand/models/topostrand/`.
- Training: `python -m methods.topostrand.run_one --stage train ...`.
- fixed3: `python -m methods.topostrand.derive_fixed3 ...`.
- Routed evaluation: `python -m methods.topostrand.evaluate_routed ...`.
- One-click: `bash scripts/run_topostrand_seeds03_200.sh`.
- Config: `configs/models/topostrand.yaml`.

## Canonical recipe

- 4 seeds, 200 epochs, 20-dataset protocol, validation-Dice selection.
- frozen SAM ViT-L + four Stage-1/Stage-2 candidates; all-dense loss,
  soft-clDice, Skeleton Recall, role losses, auxiliary fusion.
- training-only WCD; fixed3 deployment; optional 4-view flip TTA and RCD
  at inference.
- no target-domain support, masks or labels.

The final 4-seed routed table is written under
`results/runs/topostrand/final200_pa/seed_<s>/eval_routed/`.
