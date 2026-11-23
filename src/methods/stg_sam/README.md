# STG-SAM: structure-tensor-guided SAM

Independent method package built on a physical copy of the proven
GFA/BCTD-SAM scaffold. It uses only the official SAM ViT-L weights and the
fixed 3-shot protocol (18 train / 6 val / 20 test; seeds 0-3).

## Mechanism

Thin curvilinear structures are elongated; unseen-domain false positives
are usually isotropic texture blobs. STG-SAM gives the encoder an explicit,
label-free geometric evidence channel:

1. Compute the luma of the (SAM-normalized) input.
2. At three scales (sigma 1, 2, 4 at 1024), blur the luma, take image
   gradients, and form the structure tensor
   `J = [[G*(Ix^2), G*(IxIy)], [G*(IxIy), G*(Iy^2)]]`.
3. Derive coherence `C = sqrt((Jxx-Jyy)^2 + 4Jxy^2) / (Jxx+Jyy)` and the
   doubled angle `phi = atan2(2Jxy, Jxx-Jyy)`; the three features per
   scale are `C`, `C cos phi`, `C sin phi`.
4. Z-score each feature map per image (style/scale agnostic), patch-embed
   the 9-channel stack with a new 16x16-stride stem, and add it to the
   official SAM RGB patch tokens before the positional embedding.

The stem (3.1M parameters) is trained together with the per-block adapters
and the gated four-candidate decoder on the 18 source images. No target
labels, support sets, pseudo-labels, or test-time updates are used.

## Why this is a new framework

The improvement target is not another learned mask head: it is explicit
geometric evidence that cannot be inferred robustly from 18 source images.
Coherence/orientation are computed from the test image itself, so the prior
transfers across domains without target statistics.

## Ablation / matched control

`--structure-tensor 0` removes the stem and reproduces the GFA scaffold
control exactly. The 9-dataset control macro is
49.44 / 35.43 / 58.79 / 93.62 (Dice/IoU/clDice/HD95).

## Training

```bash
source scripts/wandb_env.sh
TMPDIR=/home/dev/mywork/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 \
python src/methods/stg_sam/stg_adapter.py --stage train \
  --data-root data/sacm_3shot_seed0 --workdir results/runs_local/stg_s0_200 \
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
  --structure-tensor 1 --use_wandb true
```

## Inference

```bash
PYTHONPATH=src python src/methods/stg_sam/stg_adapter.py --stage predict \
  --checkpoint results/runs_local/stg_s0_200/best_model.pth \
  --pred-root results/runs_local/stg_s0_200/predictions/stg_sam \
  --data-root data/sacm_3shot_seed0 \
  --test-datasets TYRE:datasets/novel/TYRE --device cuda:0
```

## Files

- `stg_adapter.py`: CLI entry; adds `--structure-tensor`.
- `segment_anything/modeling/image_encoder.py`: structure-tensor maps and
  the patch-embedding stem.
- `segment_anything/build_sam.py`: `use_structure_tensor` plumbing.
- `base.py`, `adapter.py`, `tcsu.py`, etc.: the copied GFA scaffold.
- Protocol, data and baseline packages are untouched.
