# SPA-SAM

SPA-SAM (Structure-Preserving Augmentation SAM) is a DCD-family baseline:

- **Architecture**: identical to DCD.  Frozen SAM ViT-L image encoder,
  Adapter-E/Fusion, four Stage-2 candidates, Stage-1 gating.
- **Objective**: identical to DCD.  Dense BCE 0.7 + Dice 0.3, Stage-1 deep
  supervision 0.3, soft-clDice 0.1, Skeleton Recall 0.1, role losses, and a
  zero-initialised linear fusion auxiliary.
- **Training change**: a structure-preserving augmentation policy.
  - `none`: no augmentation (DCD control).
  - `flip`: synchronized horizontal/vertical flips.
  - `photo`: mild brightness, contrast, gamma and Gaussian noise.
  - `mild` (default): flips + photometric.
  - `strong`: mild + random 90-degree rotations + mild blur.
  - Crops, elastic warps, cutout and mixup are excluded because they break
    thin curvilinear structures.
- **Deployment**: standard DCD fixed3 checkpoint:
  `head_select=fixed`, `deployment_head=3`, `head_fusion=0`, single-pass,
  sigmoid threshold 0.5.

## Ablations

The adapter keeps DCD's structural ablations and adds an augmentation axis:

| axis | flag(s) |
|---|---|
| augmentation | `--train-aug none|flip|photo|mild|strong` |
| specialist heads | `--specialized-heads 0` |
| fusion training | `--head-fusion 0` |
| topology losses | `--cl-dice-weight 0 --skeleton-recall-weight 0 --specialized-topology-weight 0` |
| dense scope | `--stage2-loss-scope all|selected` |
| topology scope | `--topology-head all|deployment|selected` |

`scripts/run_spa_sam_ablation.sh` runs the augmentation and structural
ablations across seeds and writes a summary.

## Local evidence

Four official splits, 50 epochs, 20-dataset macro:

| method | Dice | IoU | clDice | HD95 |
|---|---:|---:|---:|---:|
| SACM (4 seeds) | 55.40 | 41.85 | 60.34 | 67.04 |
| DCD (4 seeds) | 56.66 | 43.05 | 63.42 | 64.28 |
| SPA-SAM (4 seeds) | 57.39 | 43.76 | 66.34 | 62.95 |

200-epoch seed0/seed1: 58.44/44.74/66.30/62.07 and 58.93/44.79/67.90/61.15.
The server one-click script runs the canonical 800-epoch budget.

## Known caveats

- Seen Dice/IoU are slightly lower than DCD; clDice/HD95 are better.
- Novel HD95 is slightly worse than DCD; macro HD95 improves through Seen
  and Unseen.  TopoMortar HD95 regresses.
- Per-dataset HD95 wins are only 4/20 versus DCD even though the macro mean
  improves; report both.
- Structural decoder/prompt/self-prompt attempts are negative results and
  are not part of this baseline.

## One-click

```bash
bash scripts/run_spa_sam_seeds03_800.sh
bash scripts/run_spa_sam_ablation.sh
```
