# Post-A0 re-audit: seven fixed models

- A0: **pass** (`a0-cal-2026-10-26-v3`, `df593177d7213f6f1452f61062a41a67539acaef3009a0c43feb9e5905cedd1a`)
- A0 report: `calibration/audits/v3/calibration_report.json`
- Scope: single post-A0 **code-level** re-audit of the seven models changed in the
  fix pass. This is not a new dual audit of every model and not a main-table
  qualification. The high-risk dual-audit rule and G6c still apply.
- Pre-fix evidence remains in each `README.md`; this file supersedes only the
  gate summary for these seven models.

| model | G0 | G1 | G2 | G3 | G4 | G5 | G6a | G6b | G6c | main_table |
|---|---|---|---|---|---|---|---|---|---|---|
| `unet` | pass | pass | pass | unverified | pass | pass | pass | N/A | unverified | blocked |
| `sam_adapter` | unverified | pass | pass | unverified | pass | pass | unverified | N/A | unverified | blocked |
| `knowsam` | pass | pass | unverified | unverified | unverified | pass | unverified | N/A | unverified | blocked |
| `matcher` | pass | unverified | unverified | unverified | unverified | pass | unverified | N/A | unverified | blocked |
| `ucs` | pass | pass | pass | unverified | pass | pass | unverified | N/A | unverified | blocked |
| `nnwnet` | pass | pass | unverified | unverified | pass | pass | unverified | N/A | unverified | blocked |
| `persam` | pass | pass | pass | pass | pass | pass | unverified | N/A | unverified | blocked |

## Per-model result

### `unet`

- Resolved/disclosed: D-unet-03 (global Dice), D-unet-09 (workers/RNG disclosed), D-unet-13 (UPSTREAM path).
- Why not main table: D-unet-03 fixed; G6a 20-step probe loss/grad/param diff=0; G6c seed0 full run + repeated eval identical (3580 masks); seeds 1–4 stopped on the 11 GiB host. G3 recipe evidence and seed variance remain.
- Claim label: `none`

### `sam_adapter`

- Resolved/disclosed: D-05 (weight decay), D-03/D-04/D-06 (disclosure).
- Why not main table: D-05 fixed (wd=0.01); 4-GPU/batch/workers/seed disclosed. G0 has no usable commit/license; G3 lacks G6a.
- Claim label: `none`

### `knowsam`

- Resolved/disclosed: D-kn-02 (test exclusion), D-kn-04 (largest-CC), D-kn-05 (argmax/scipy zoom), D-kn-03/06 (disclosure).
- Why not main table: Test leakage fixed; only 3 domains remain eligible, disclosed. G2/G3/G4 lack post-fix probes; G6c unrun.
- Claim label: `none`

### `matcher`

- Resolved/disclosed: D-mat-01 (FSS 0.8/0.2/1.0), D-mat-02 (prediction seed flag).
- Why not main table: FSS score and prediction seed fixed in code, but DINOv2 weights are missing; G4 cannot be numerically verified. Empty-support path still undisclosed.
- Claim label: `none`

### `ucs`

- Resolved/disclosed: D-ucs-06 (initial prompt seeding), D-ucs-07/D-ucs-10 (disclosure).
- Why not main table: Prediction prompt seeding fixed; workers/metrics disclosed. G3 lacks training/G6a; G6c unrun. Oracle/reference only.
- Claim label: `none`

### `nnwnet`

- Resolved/disclosed: D-nnw-03 (val leakage), D-nnw-05 (actual batch metadata), D-nnw-11 (G0 URL/commit).
- Why not main table: Train-only planning and metadata fixed with tests; full nnU-Net run still needed. G2/G3 lack loss/grad and G6a; G6c unrun.
- Claim label: `none`

### `persam`

- Resolved/disclosed: D-persam-01 (raw vs normalized reference embedding split).
- Why not main table: D-persam-01 fixed; final-mask G2 remains pass. G6a must be re-run after the first-stage change; G6c unrun.
- Claim label: `none`

## What is still missing

- `matcher`: DINOv2 ViT-L/14 weights; G1/G2/G4 remain unverified.
- `sam_adapter`: usable upstream commit/license for G0; G3 still has recipe/runtime deviations.
- `knowsam`: post-fix G2/G3/G4 probes; 384-cap coverage is only CHASEDB1/CREMI/CORN.
- `nnwnet`: full nnU-Net planning/preprocessing/training run; G2/G3 probes.
- `ucs`: training/G6a; official metric set is not reproduced.
- `persam`: post-fix G6a re-run.
- All: G6c full training/eval for a main-table claim.
