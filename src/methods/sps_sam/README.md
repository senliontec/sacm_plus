# SP-SAM: self-prompting SAM for curvilinear structures

Independent method package. It uses only the official SAM ViT-L weights
(`checkpoints/sam_vit_l_0b3195.pth`) and the fixed 3-shot protocol
(18 train / 6 val / 20 test images; seeds 0-3). It does not import any
baseline model package and does not modify the data or the protocol.

## Why this framework

The zero-shot diagnostic in `results/reports/iccv_model_search_2026-11-21.md`
shows that the official frozen SAM promptable decoder returns large regions
on curvilinear structures even with GT clicks (DRIVE 0.21 Dice, ROAD 0.01,
TYRE 0.00, XCAD 0.04 for the best of three masks). Prompt ensembling of the
frozen decoder is therefore not viable. SP-SAM instead *trains* the
promptable decoder to return the local curvilinear structure around a click
and *predicts the clicks*.

## Mechanisms

1. **Prompt proposal.** A head upsamples the 64x64 SAM embedding to a
   256x256 click heatmap, supervised by the GT skeleton. At inference the
   top peaks (default K=64, 2-cell NMS on the 256 grid) become point
   prompts. No target labels are used.
2. **Point-conditioned decoding.** The official SAM prompt encoder and
   two-way transformer decode mask token 0 for each click. The target is
   the GT inside a 128-pixel disk centred at the click (a local structure
   patch); clicks far from any structure have an empty target. Full
   connected-component targets were tried first and caused prompt collapse:
   every foreground click on one-component GT had the same target, so the
   decoder ignored the click.
3. **Weighted-union aggregation.** Per-click local patches are combined by
   `max_i(sigmoid(iou_i) * p_i)` after normalising the weights by the best
   click. A plain mean would erase patches covered by a minority of clicks.
   Training supervises the aggregate against the union of the local targets,
   not the full GT, so the model is never asked to hallucinate uncovered
   regions. No hand guard, consensus gate, TCSU residual, snake upsampling,
   or component-size filter is used.

Only the per-block SAM adapters, the prompt encoder, the official stage-1
mask-decoder weights and the new proposal head are trained (~10M of 324M
parameters). The encoder blocks start frozen from official SAM.

## Training

```bash
source scripts/wandb_env.sh
TMPDIR=/home/dev/mywork/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 \
python src/methods/sps_sam/adapter.py --stage train \
  --data-root data/sacm_3shot_seed0 --workdir results/runs_local/sps_s0_200 \
  --epochs 200 --seed 0 --device cuda:0 \
  --lr 3e-4 --grad-clip 1.0 --scheduler cosine --warmup-epochs 5 \
  --batch-size 1 --num-workers 0 --save-every 0 --use_wandb true
```

Validation uses the 6-image Seen split and selects the best epoch by Dice,
as in the shared protocol. Augmentation is structure-preserving: flips,
90-degree rotations, mild photometric jitter and noise.

## Inference / metrics

```bash
PYTHONPATH=src python src/methods/sps_sam/adapter.py --stage predict \
  --checkpoint results/runs_local/sps_s0_200/best_model.pth \
  --pred-root results/runs_local/sps_s0_200/predictions/sps_sam \
  --data-root data/sacm_3shot_seed0 \
  --test-datasets TYRE:data/sacm_3shot_seed0/test/TYRE \
  --device cuda:0
```

The script writes a probability map per image at GT resolution; the shared
runtime thresholds at 0.5 for metrics. Dataset-equal macro Dice/IoU/clDice/
HD95 are computed by the shared evaluator.

## Key flags

| flag | default | meaning |
|---|---:|---|
| `--sp-train-k` | 8 | clicks per training image |
| `--sp-neg-k` | 2 | background clicks (empty local targets) |
| `--sp-deploy-k` | 64 | clicks at inference |
| `--sp-nms-radius` | 2 | NMS distance in 256x256 cells |
| `--sp-local-radius` | 128 | per-click target disk radius (1024 frame) |
| `--sp-min-click-dist` | 64 | minimum spacing of structure clicks |
| `--sp-point-weight` | 0.5 | per-click mask loss |
| `--sp-agg-weight` | 1.0 | aggregated-mask loss |
| `--sp-cl-weight` | 0.1 | soft-clDice on the aggregate |
| `--sp-adapters` | 1 | train the encoder adapters (0 = frozen encoder) |
| `--sp-prompt-encoder` | 1 | train the official prompt encoder |

## Protocol and caveats

- Source-only training; target images are used only for forward inference
  and metrics. No target labels, support sets, pseudo-labels, transductive
  steps, gradient updates, or test-time adaptation.
- Images are resized to 1024x1024 for this screen (no tiling). This can
  lose thin structures on high-resolution datasets; tiling is a planned
  follow-up, not part of the screen.
- GT masks are bilinearly resized and thresholded at 0.25 to preserve thin
  lines; training targets are max-pooled to the 512 loss resolution.
- All reported constants (K, NMS radius, thresholds) are fixed, not tuned on
  test. Any later change selected on val must be disclosed.

## Files

- `model.py`: proposal head, official prompt-decoder math, model wrapper,
  peak selection.
- `adapter.py`: runtime adapter (training, losses, inference), CLI entry.
- `segment_anything/`: self-contained copy of the SAM code; no baseline
  model import.
- `topology_losses.py`: soft-clDice used on the aggregated mask.
- `tests/test_sps_sam.py`: fast unit tests.
