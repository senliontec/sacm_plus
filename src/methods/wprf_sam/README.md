# WPRF-SAM: widest-path reachability for SAM curvilinear decoding

Independent method package. It keeps the proven GFA/BCTD-SAM scaffold
(copied physically; official SAM ViT-L weights only) and adds the
Widest-Path Reachability Field training objective from

> Widest-Path Reachability Fields for Connectivity-Preserving Slender
> Structure Segmentation (2026), found in the local parsing corpus.

The paper's code link (`github.com/mituan-ai/WPRF`) is 404, so this is an
implementation from the paper.

## Mechanism

1. An **affinity head** on the SAM encoder embedding predicts directed
   8-neighbour link logits on a 128x128 graph grid (stride 8 at 1024).
2. A **support graph** is built from the GT skeleton (max-pooled to the
   graph grid, 8-connected components). Dense foreground components
   provide edge labels.
3. A differentiable **Max-Min DP** computes k-step widest-path
   reachability `r_k(u,v) = max_path min_edge w` for scales
   `{1,2,4,8,16}`; `L_edge` and `L_reach` are BCE terms on the
   symmetrised links and sampled reachability pairs.
4. Gradients concentrate on the weakest edge of a path, repairing
   connectivity bottlenecks. Inference emits only the union mask, so the
   head costs nothing at test time.

Only the affinity head (and, through the shared embedding, the adapters and
decoder) is trained; the official SAM encoder blocks stay frozen.

## Deviations from the paper

- Graph stride 8 at 1024 input (paper: 4 at 512); 16 sources per image
  (paper: 32).
- The bottleneck-aware segmentation term is omitted because the scaffold
  already trains BCE/Dice, soft-clDice and skeleton recall.
- The affinity head starts with a positive output bias and the BCE terms
  are clamped to keep the Max-Min DP numerically alive at step 0.
- Edge weights default to 1.0/1.0; a second variant uses 0.5/0.5 with a
  10-epoch warmup, and a reach-only ablation uses edge 0.0 / reach 0.5.

## Training

```bash
source scripts/wandb_env.sh
TMPDIR=/home/dev/mywork/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 \
python src/methods/wprf_sam/wprf_adapter.py --stage train \
  --data-root data/sacm_3shot_seed0 --workdir results/runs_local/wprf_s0_200 \
  --epochs 200 --seed 0 --device cuda:0 --lr 3e-4 \
  --bce-weight 0.7 --dice-weight 0.3 --cl-dice-weight 0.1 \
  --skeleton-recall-weight 0.1 --boundary-loss-weight 0.05 \
  --boundary-loss-resolution 512 --boundary-loss-head boundary \
  --boundary-loss-warmup-epochs 5 --area-loss-weight 0.1 \
  --area-loss-mode two_sided --deployed-tversky-weight 0.1 \
  --specialized-heads 1 --deployment-head 3 --topology-head all \
  --topology-warmup-epochs 5 --head-fusion 1 --head-select gating \
  --fusion-loss-weight 1.0 --copy-paste-prob 0.5 --train-aug mild \
  --cosna-train-scope full --cosna-weight 0.0 --cosna-cl 0.0 \
  --cosna-render 1024 --cosna-coarse 512 --cosna-feature-dim 16 \
  --cosna-gate 0.10 --cosna-near-radius 4 \
  --cosna-filter-min-size 80 --cosna-filter-aspect 2 \
  --cosna-filter-maxdim 1200 --rcd 1 --as-calibration 1 \
  --close-radius 2 --tile 1024 --stride 896 \
  --routing-skel-ratio 1.0 --routing-area-ratio 0.7 \
  --routing-mode auto --num-workers 0 --save-every 0 \
  --wprf 1 --wprf-edge-weight 1.0 --wprf-reach-weight 1.0 \
  --wprf-stride 8 --wprf-sources 16 --use_wandb true
```

## Inference

```bash
PYTHONPATH=src python src/methods/wprf_sam/wprf_adapter.py --stage predict \
  --checkpoint results/runs_local/wprf_s0_200/best_model.pth \
  --pred-root results/runs_local/wprf_s0_200/predictions/wprf_sam \
  --data-root data/sacm_3shot_seed0 \
  --test-datasets TYRE:datasets/novel/TYRE --device cuda:0
```

## Current evidence (seed 0)

- 20 datasets: 62.14 / 48.63 / 69.79 / 59.06
  (CoSNA 4-seed: 62.09 / 48.42 / 70.21 / 59.07) -> 3/4, clDice -0.42.
- 12 subset: 64.55 / 51.59 / 73.40 / 57.66
  (CoSNA: 64.61 / 51.40 / 73.63 / 56.99) -> 2/4.
- Strong gains on fragile unseen data (DeepCrack, FIVES, CHCrack5k,
  LEAF, ROAD, TYRE, STARE, LES-AV), but TopoMortar collapses to almost
  empty on about 25% of images (raw Stage-2 probability ~0.004 vs the
  control's 0.063 on a failing case).  The 0.5/0.5 + 10-epoch warmup
  variant and the reach-only ablation target this failure.

## Files

- `wprf.py`: affinity head, support graph, Max-Min DP, edge/reach losses.
- `wprf_adapter.py`: CLI entry; WPRF flags.
- `base.py`: cached encoder embedding + WPRF loss hook with warmup.
- `segment_anything/`, `adapter.py`, `tcsu.py`, ...: copied GFA scaffold.
