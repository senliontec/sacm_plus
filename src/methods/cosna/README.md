# CoSNA: Consensus-Guided Snake Reconstruction

Independent 3-shot curvilinear segmentation package. CoSNA keeps the frozen
SAM ViT-L + four-candidate BCTD scaffold, and adds:

1. **Gated candidate consensus** — a label-free mask over the four Stage-2
   candidates: where candidate disagreement is low, use h3; where it is high,
   require all four heads to agree.
2. **TCSU-lite reconstruction** — a zero-initialized, image-guided snake
   upsampling residual on candidate 3. Only the 18k-parameter residual is
   trained on the 18 source images; the base is frozen.
3. **Near-union** — add TCSU-h3 pixels only within 4 px of the gated mask,
   then run close2 + Adaptive Skeleton Calibration (AS).

The package owns its BCTD model and adapter code under
`src/methods/cosna/models/cosna` and `src/methods/cosna/base.py`; it does not
import the baseline BCTD model/adapter at runtime.

## Layout

```
src/methods/cosna/
  adapter.py        CoSNAAdapter: train TCSU and predict CoSNA
  tcsu.py           StructuralEvidence / SnakeUpsampler / TCSU-lite
  base.py           physically copied BCTD adapter
  models/cosna/     physically copied BCTD SAM model package
  run_one.py        python -m methods.cosna.run_one entrypoint
configs/models/cosna.yaml
scripts/run_cosna_seeds03.sh
```

## Initialization

The BCTD base must start from a trained BCTD checkpoint. Copy the champion
checkpoint into a new CoSNA workdir as `best_model.pth` (the runtime's resume
path), then train the TCSU head for 20 epochs:

```bash
# seed0 champion
cp results/runs/topostrand/pilot200_bnd005_h2/seed_0/best_model_fixed3.pth \
   results/runs/cosna/seed_0/best_model.pth

# seeds1-3
for s in 1 2 3; do
  mkdir -p results/runs/cosna/seed_$s
  cp results/runs/topostrand/bctd200/seed_$s/best_model_fixed3.pth \
     results/runs/cosna/seed_$s/best_model.pth
done
```

If the copied checkpoint reports a data-manifest mismatch caused only by the
absolute-vs-relative path convention, repatch the copied checkpoint's
`data_manifest_sha256` with the current data-root fingerprint; never modify
the original champion checkpoint. The one-click script below handles this.

## Train

```bash
PYTHONPATH=src python -m methods.cosna.run_one --stage train \
  --data-root data/sacm_3shot_seed0 \
  --workdir results/runs/cosna/seed_0 \
  --epochs 96 --seed 0 --device cuda:0 --use_wandb 0 \
  --resume --resume-checkpoint best --lr 1e-3 \
  --cosna-weight 1.0 --cosna-cl 0.1 \
  --cosna-render 1024 --cosna-coarse 512 \
  --cosna-gate 0.10 --cosna-near-radius 4
```

`--epochs` is the absolute final epoch. If the copied BCTD checkpoint is at
epoch 76, use 96 for 20 TCSU epochs. The base is frozen by
`configure_trainable`; only `model.tcsu` receives gradients.

## Predict

```bash
PYTHONPATH=src python -m methods.cosna.run_one --stage predict \
  --data-root data/sacm_3shot_seed0 \
  --protocol data/sacm_3shot_seed0/protocol_manifest.csv \
  --workdir results/runs/cosna/seed_0 \
  --checkpoint results/runs/cosna/seed_0/best_model.pth \
  --seed 0 --device cuda:0 --use_wandb 0 \
  --rcd 1 --as-calibration 1 --close-radius 2 \
  --tile 1024 --stride 896 --routing-skel-ratio 1.0 \
  --routing-area-ratio 0.7 --routing-mode auto
```

`predict_sample` implements four-view flip TTA, full-vs-tiled RCD routing,
close2, AS and the 4 px near-union. No target labels or target adaptation
are used.

## One-click

```bash
bash scripts/run_cosna_seeds03.sh
```

The script copies the appropriate BCTD champion checkpoint per seed, patches
only the copied checkpoint's manifest hash, trains the TCSU head for 20
epochs, and runs prediction for the requested seeds. It does not modify
baseline code or the original champion checkpoints.

## Reference result (seed0-3, 20 datasets)

| | Dice | IoU | clDice | HD95 |
|---|---:|---:|---:|---:|
| CoSNA 4-seed macro | 62.04 | 48.38 | 70.28 | 59.22 |
| BCTD-SAM + AS champion | 61.93 | 48.29 | 70.14 | 59.23 |

CoSNA improves all four 4-seed macro metrics, but seed2/3 HD95 is 0.29/0.33
higher; it is a rule-level near-miss, not the current champion. See
`results/reports/night_2026-11-20.md`.

## Caveats

- CoSNA is an inference-time two-view fusion of the gated and TCSU branches;
  the TCSU branch is trained source-only.
- The 4 px radius and 0.10 gate are label-free but selected on Seen
  validation; validate on a held-out split before a paper claim.
- Do not mix this 20-epoch head fine-tune with 200-epoch from-scratch runs.
- No target labels, supports, pseudo-labels, or test-time adaptation.
