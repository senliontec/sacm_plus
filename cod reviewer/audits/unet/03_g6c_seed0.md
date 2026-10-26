# U-Net G6c seed 0 pilot

- Gate: G6c
- Status: **partial_seed0** (one seed; seed variance still needed)
- Device: cuda:0
- Training: 200 epochs; best epoch 195; best val Dice 0.390508
- W&B train: http://172.16.1.7:8080/buaazqk/sacm/runs/cxnfs6r3
- W&B paper metrics: http://172.16.1.7:8080/buaazqk/sacm/runs/zs9plt3y
- Checkpoint: `results/paper/unet/seed_0/best_model.pth`
- Checkpoint SHA-256: `d1f191eeb7406314d4940290d0f94011df8cccd160212fb930dc83f9d553ae7a`
- Split manifest SHA-256: `d5e5293359bb17c02a61a8dba64c61187c90e02d7937e3aba3e45a6b09e30888`

## Full evaluation

- 20 test datasets, 3580 masks saved under `results/paper/unet/seed_0/predictions/unet`.
- Macro All: Dice 0.287775, IoU 0.205348,
  clDice 0.313281, HD95 131.329.

## Independent eval repeat

- Re-ran `run_baseline_test.py` with the same checkpoint into
  `results/paper/unet/seed_0/predictions_repeat/unet` and `repeat/`.
- 3580/3580 mask files present; SHA-256 byte differences: **0**.
- `paper_metrics.csv` identical: **yes**.

## Why G6c is not pass yet

G6c requires seed mean/variance. This pilot covers one seed and an independent
eval repeat only. Seeds 1–4, their full train/eval runs, and recomputed
variance are still pending. This is not a main-table claim.
