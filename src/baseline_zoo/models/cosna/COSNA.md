# CoSNA baseline package

CoSNA = Consensus-Guided Snake Reconstruction. This directory is a
self-contained baseline copy of the model. It owns the SAM ViT-L scaffold,
four candidate heads, TCSU-lite residual, decoding, losses and
adaptive-skeleton calibration. It does not import `methods.cosna`, the BCTD
model package, or another model package at runtime.

## Mechanism

1. Four-view flip TTA over the four Stage-2 candidates.
2. Gated candidate-consensus mask: low-disagreement pixels use candidate 3;
   high-disagreement pixels require all four candidates to agree.
3. TCSU-lite snake-upsampling residual on candidate 3: 18k parameters,
   zero-initialized, trained on the 18 source images while the scaffold is
   frozen.
4. Constrained near-union: add TCSU pixels within 4 px of the gated mask,
   then close2 + adaptive-skeleton calibration.
5. Resolution-adaptive added-component filter: remove added components
   below 80 px; on images with max dimension >=1200 retain components with
   bbox aspect >=2.

## Reproduction

The canonical CoSNA checkpoints are the four TCSU runs in
`results/runs/bctd/tcsu20/seed_{0..3}/best_model.pth`. They carry the BCTD
scaffold weights plus the trained TCSU residual. The baseline runtime
requires the copy's adapter tag to be `cosna`; `prepare_init` changes only
the copy.

Quick reproduction of all four seeds and 20 datasets:

```bash
bash scripts/run_cosna_baseline_seeds03.sh
```

This is `MODE=repro` by default. It copies each canonical checkpoint to
`results/runs/cosna/default/seed_X/best_model.pth`, patches only the copy,
then runs the standard baseline pipeline. Tables land in
`results/runs/cosna/default/seed_X/eval/tables/all_metrics.csv`.

Train a fresh model from the BCTD champion checkpoints:

```bash
MODE=train bash scripts/run_cosna_baseline_seeds03.sh
```

Server one-click (GPUs 0-3, seeds 0-3, train then 20-dataset eval and
4-seed summary):

```bash
bash scripts/run_cosna_server_seeds03.sh
```

The server wrapper uses `results/runs/cosna/server_scratch` for `INIT=scratch`
(and `server_cosna`/`server` for the other modes), writes per-seed canonical
tables, then `results/reports/cosna_server_4seed/summary.{json,csv,md}`.
`EPOCHS` defaults to `800` and is an absolute final-epoch target.

Initialization modes:

```bash
# Default: fully independent of BCTD and CoSNA init weights.
# Train scaffold adapters/decoder + TCSU jointly from SAM ViT-L only.
bash scripts/run_cosna_server_seeds03.sh

# Optional: initialize from CoSNA-owned checkpoints (not BCTD).
INIT=cosna COSNA_INIT_ROOT=checkpoints/cosna_init \
  bash scripts/run_cosna_server_seeds03.sh
```

`INIT=scratch` reads only `checkpoints/sam_vit_l_0b3195.pth`. `INIT=cosna`
expects `COSNA_INIT_ROOT/seed_<n>/best_model.pth`; it is useful when the
canonical CoSNA initialization weights are available. For that optional
mode, place the four CoSNA checkpoints there explicitly:

```bash
mkdir -p checkpoints/cosna_init
# for seed in 0 1 2 3: copy a CoSNA checkpoint to
# checkpoints/cosna_init/seed_<seed>/best_model.pth
```

`INIT=bctd` is the legacy local path and is not required on the server.

For the canonical 20-TCSU-epoch reproduction use the lower-level script
(`MODE=train`, default `EPOCHS_ADD=20`) with an available initialization
checkpoint. Use `DRY_RUN=1` to inspect commands.

`EPOCHS_ADD=20` is the number of TCSU epochs, not an absolute epoch. The
source-mode wrapper computes the absolute final epoch from the initialization
checkpoint. The YAML config's own train entrypoint is independent scratch at
`defaults.epochs=800`; use the lower-level source script for the canonical
20-epoch recipe.

Run one seed manually from scratch (no BCTD/CoSNA init checkpoint):

```bash
python -m baseline_zoo.models.cosna.train \
  --init-mode scratch --epochs-total 800 \
  --data-root data/sacm_3shot_seed0 \
  --workdir results/runs/cosna/default/seed_0 \
  --seed 0 --device cuda:0 --use_wandb 1 \
  --sam-checkpoint checkpoints/sam_vit_l_0b3195.pth \
  --lr 1e-3 --cosna-weight 1.0 --cosna-cl 0.1 \
  --cosna-render 1024 --cosna-coarse 512 --cosna-feature-dim 16 \
  --cosna-gate 0.10 --cosna-near-radius 4 \
  --cosna-filter-min-size 80 --cosna-filter-aspect 2 \
  --cosna-filter-maxdim 1200 --rcd 1 --as-calibration 1 \
  --close-radius 2 --tile 1024 --stride 896 \
  --routing-skel-ratio 1.0 --routing-area-ratio 0.7 \
  --routing-mode auto --num-workers 0 --save-every 0

python scripts/pipeline/run_baseline_test.py --method cosna \
  --protocol data/sacm_3shot_seed0/protocol_manifest.csv --seed 0 \
  --checkpoint results/runs/cosna/default/seed_0/best_model.pth \
  --data-root data/sacm_3shot_seed0 \
  --pred-root results/runs/cosna/default/seed_0/predictions \
  --paper-root results/runs/cosna/default/seed_0/eval \
  --device cuda:0 --use_wandb false --skip-visualize
```

For a validation split or the exact JSON summary used in the port audit:

```bash
python -m baseline_zoo.models.cosna.evaluate \
  --data-root data/sacm_3shot_seed0 \
  --workdir results/runs/cosna/default/seed_0 \
  --checkpoint results/runs/cosna/default/seed_0/best_model.pth \
  --split val --device cuda:0 \
  --out results/runs/cosna/default/seed_0/val_summary.json
```

## Reference result (4 seeds, dataset-equal macro)

| | Dice | IoU | clDice | HD95 |
|---|---:|---:|---:|---:|
| CoSNA 20 datasets | 62.09 | 48.42 | 70.21 | 59.07 |
| BCTD-SAM + AS 20 datasets | 61.93 | 48.29 | 70.14 | 59.23 |
| CoSNA 12-dataset subset | 64.61 | 51.40 | 73.63 | 56.99 |
| BCTD-SAM + AS 12-dataset subset | 64.57 | 51.37 | 73.63 | 57.13 |

Exact 12-subset: `64.61125 / 51.395625 / 73.629792 / 56.98714`, all four
strictly above BCTD `64.5742 / 51.3682 / 73.6274 / 57.1322`; clDice margin
is +0.0024.

## Port verification

- Full seed0, all 20 test datasets (3580 images, 0 missing): every
  per-dataset metric matches the canonical CoSNA m=80 result to below 1e-4.
  Macro `63.2135 / 49.5385 / 71.0880 / 57.7632` versus canonical
  `63.21 / 49.54 / 71.09 / 57.76`.
- Four-seed validation matches exactly: `0.6619/0.5130/0.7370/11.6825`,
  `0.6651/0.5161/0.7368/7.5134`, `0.6818/0.5369/0.7610/6.7952`,
  `0.6661/0.5160/0.7248/9.7873`.
- DRIVE test `0.8023/0.6707/0.8295/6.0888` with byte-identical masks;
  DeepCrack `0.7372/0.6023/0.8169/45.8443`; HRF (large/tiled path)
  `0.7917/0.6560/0.8062/33.5681`.
- The standard baseline pipeline writes DRIVE `dice_mean=0.80233266`,
  `iou_mean=0.67067750`, `cldice_mean=0.82946516`,
  `hd95_mean=6.08881812`, matching the canonical values.
- Evidence: `results/reports/cosna_baseline_port_audit_2026-11-20.md` and
  `results/ablations/cosna_m80/baseline_port/`.

## Caveats

- The 80 px threshold was selected at the 12-dataset test gate after m=100
  missed clDice by 0.0032; validation is identical for m=80 and m=100, so
  the 12-subset pass is benchmark-selected, not independent confirmation.
- CoSNA is a two-branch inference method (gated + TCSU-h3), not a single
  end-to-end decoder pass.
- The canonical TCSU head is a 20-epoch source-only fine-tune of a frozen
  scaffold; do not mix with 200-epoch from-scratch runs.
- `--init-mode scratch` trains scaffold adapters/decoder plus TCSU jointly
  from SAM ViT-L. It is independent of BCTD weights, but its results are not
  the canonical 20-epoch CoSNA numbers and need a fresh 4-seed evaluation.
- `INIT=cosna` needs CoSNA-owned init checkpoints under `COSNA_INIT_ROOT`;
  it never reads a BCTD path.
- No target labels, target supports, pseudo-labels, or test-time adaptation.
