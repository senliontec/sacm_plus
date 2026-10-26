# All-model provisional audit summary

> A0 = unverified for every report (no frozen calibration pack). No model is main-table eligible.
> G6b is N/A because no compatible official method checkpoint was used. G6c is unverified for all models.
> `pass` means the stated gate has evidence in that model report; it does not mean official metric reproduction.

> **Post-fix code pass (2026-10-26):** seven confirmed code defects were fixed and eleven
> missing deviation disclosures were added; 13 `UPSTREAM.md` provenance records
> were corrected. The gate table below is intentionally the pre-fix audit snapshot.
> Current code-level status and evidence: [`POST_FIX.md`](POST_FIX.md).

| model | G0 | G1 | G2 | G3 | G4 | G5 | G6a | G6b | G6c | main blocker |
|---|---|---|---|---|---|---|---|---|---|---|
| `unet` | pass | pass | fail | unverified | pass | pass | unverified | N/A | unverified | G2: official global Dice vs port per-image Dice |
| `cs2net` | pass | pass | pass | unverified | pass | pass | pass | N/A | unverified | G3: single-GPU vs 2-GPU DataParallel; workers/RNG undocumented |
| `bcunet` | pass | unverified | unverified | unverified | unverified | pass | unverified | N/A | unverified | official model is not constructible (SyntaxError/upconv2x2) |
| `maskvsc` | pass | pass | pass | unverified | pass | pass | unverified | N/A | unverified | G3: ratio=0 RNG, flip/worker differences |
| `nnwnet` | pass | pass | unverified | unverified | pass | fail | unverified | N/A | unverified | G5: val images enter planner/normalization before split |
| `jtfn` | pass | pass | pass | unverified | pass | pass | unverified | N/A | unverified | G3: per-batch StepLR vs per-epoch; augmentation/workers |
| `dscnet` | pass | pass | pass | unverified | pass | pass | unverified | N/A | unverified | G3: train/test mean/std, MONAI->PIL, seed/worker |
| `segdino` | pass | unverified | unverified | unverified | unverified | pass | unverified | N/A | unverified | official DINOv3 backbone absent; head-only evidence |
| `sam_med2d` | pass | pass | pass | unverified | pass | pass | unverified | N/A | unverified | G3: workers/RNG; decoder-head inheritance differs |
| `sam_octa` | pass | pass | pass | unverified | unverified | pass | unverified | N/A | unverified | G4: official ViT-H/class-conditional clDice/metrics not preserved |
| `cwsam` | pass | pass | pass | unverified | unverified | pass | unverified | N/A | unverified | G4: official 5-class weighted CE/ViT-B/metrics differ |
| `sam_adapter` | unverified | pass | pass | unverified | pass | pass | unverified | N/A | unverified | G3: official AdamW wd=0.01 vs port 0.0; 4-GPU/batch2 |
| `topolora` | pass | pass | pass | unverified | pass | pass | pass | N/A | unverified | G3: runtime partial-accumulation rescale/param filtering undocumented |
| `samed` | pass | pass | pass | unverified | pass | pass | pass | N/A | unverified | G3: official default SGD vs README AdamW; DataParallel/RNG |
| `fob_sam` | pass | pass | pass | unverified | pass | fail | pass | N/A | unverified | G5: official test support from held-out volume; 3D supervoxel vs 2D GT |
| `sacm_official` | unverified | pass | pass | unverified | pass | fail | pass | N/A | unverified | G5: false 1024-eval claim; official resizes to original; G0 no commit/LICENSE |
| `persam` | pass | pass | pass | pass | pass | pass | pass | N/A | unverified | G6c only; D-persam-01 first-pass normalization undocumented |
| `matcher` | pass | unverified | unverified | unverified | fail | pass | unverified | N/A | unverified | G4: FSS alpha/beta/exp 0.8/0.2/1 vs 1/0/0; unseeded; DINOv2 missing |
| `knowsam` | pass | pass | unverified | fail | fail | fail | unverified | N/A | unverified | test leakage: 166/384 unlabeled images are exact test images |
| `sam2_unet` | pass | pass | unverified | unverified | pass | pass | unverified | N/A | unverified | missing SAM2 Hiera-L weights; no same-state parity |
| `sam_vit_l_unet` | pass | pass | unverified | unverified | pass | pass | unverified | N/A | unverified | no official decoder checkpoint; full run failed paper_metrics |
| `ucs` | pass | pass | pass | unverified | pass | pass | unverified | N/A | unverified | oracle-prompted; eval prompt sampling unseeded; no full G6a/G6c |

## Counts

- G2: 14 pass, 1 fail (`unet`), 7 unverified
- G3: 1 pass (`persam`), 1 fail (`knowsam`), 20 unverified
- G4: 16 pass, 2 fail (`matcher`, `knowsam`), 4 unverified
- G5: 18 pass, 4 fail (`nnwnet`, `fob_sam`, `sacm_official`, `knowsam`)
- G6a: 6 pass (`cs2net`, `topolora`, `samed`, `fob_sam`, `sacm_official`, `persam`), 16 unverified
- main_table: 0 pass; 22 blocked

## Highest-priority defects

1. **KnowSAM test leakage**: 166/384 unlabeled images are exact Seen test images (DRIVE 16/16, CHASEDB1 20/20, DCA1 130/130); image+mask pixel-equal. Fix the unlabeled pool exclusion before any result is usable.
2. **U-Net loss mismatch**: official Dice reduces globally over B,H,W (`reduce_batch_first=True`); port averages per-image Dice. Heterogeneous batch-4 probe: loss diff 0.1120, grad diff 0.00726 → `G2=fail`.
3. **SAM-Adapter weight decay**: official `AdamW` receives only `lr`, so PyTorch default `weight_decay=0.01` is active; port uses 0.0 and `UPSTREAM.md` incorrectly says no weight decay.
4. **Matcher FSS recipe**: official FSS script uses `alpha=0.8, beta=0.2, exp=1.0`; port uses `1.0/0.0/0.0`, and prediction is unseeded. `G4=fail`.
5. **nnWNet val leakage**: nnU-Net planning/preprocessing runs before `splits_final.json` is written, so the 6 val cases affect normalization/plans; `G5=fail`.
6. **SACM official documentation**: adapter claims upstream evaluates at 1024, but official `test_sam.py:260-261` resizes predictions to original size; G5 fail and G0 missing commit/LICENSE.
7. **FoB-SAM support protocol**: official test support is a held-out volume inside the test fold and only labeled query slices are evaluated; port uses training/donor support and all query images; `G5=fail`.
8. **SAMed recipe ambiguity**: official source defaults to SGD unless `--AdamW` is passed; README command uses `--AdamW`. The port uses AdamW but does not resolve/disclose the source-vs-README ambiguity; `G3=unverified`.
9. **TopoLoRA shared-runtime difference**: runtime rescales the final partial accumulation group and filters optimizer parameters; not in adapter deviations; `G3=unverified`.

## What is actually established

- Model/state_dict/forward parity has evidence for 14 models, but several probes are scoped: `sam_octa`/`cwsam` use reduced-width or 2-class setups, and `sam_med2d` has decoder-head inheritance differences. `G2=pass` is therefore not a full official-config parity claim.
- This establishes **implementation/path equivalence**, not official checkpoint reproduction or official paper metrics.
- No model has G6c full training/eval; no model can be placed in the main table until A0 and G6c are completed.


## Post-fix code status

These are code-level outcomes, not re-audited gate results. A0 and G6c remain unverified for every model, so the main table remains blocked.

| model | post-fix | evidence |
|---|---|---|
| `unet` | fixed G2 global Dice | `03_numeric_probe_after_fix.json`: loss/grad diff 0.0; `test_unet.py` |
| `sam_adapter` | fixed weight decay = 0.01; disclosed 4-GPU/workers/val selection | `test_sam_adapter.py`; `UPSTREAM.md` |
| `knowsam` | fixed test-pool exclusion, largest-CC, argmax/scipy zoom order=0; after exclusion only CHASEDB1/CREMI/CORN remain, so 384-cap coverage is limited and disclosed | leak and CC tests in `test_runtime_fixes.py` |
| `matcher` | fixed FSS 0.8/0.2/1.0 and prediction seed flag | `test_matcher.py` default/seed tests |
| `ucs` | fixed deterministic prediction prompt sampling; disclosed workers/metrics | `test_ucs.py` seed test |
| `nnwnet` | fixed train-only planner/preprocess and W&B batch metadata | `test_nnwnet_planning_stage_excludes_val` |
| `persam` | fixed raw-vs-normalized reference embedding split | `test_persam.py` embedding split/raw-prompting tests |
| `sacm_official` | corrected false 1024-eval claim; commit/license/split/workers documented | `UPSTREAM.md`; deviations |
| `sam_octa` | class-conditional clDice and metric/seed differences disclosed; still not reproduced | deviations |
| `fob_sam` | official support protocol disclosed; **G5 remains fail** | deviations |
| docs-only | `cs2net`, `jtfn`, `dscnet`, `maskvsc`, `sam_med2d`, `topolora`, `samed` received missing worker/RNG/accumulation disclosures | deviations |
| G0 | stale `baselines/...` paths and missing commits fixed for 13 UPSTREAM records | `POST_FIX.md` |

Still blocking: A0 calibration, G6c for all models, G6a for most models, missing DINOv2/SAM2 checkpoints, and protocol gaps such as FoB-SAM support semantics. The fixed models (including `persam`) still need a 02-style re-audit; a code fix is not a gate pass.

## Reports

Per-model reports: `cod reviewer/audits/<model>/README.md`; numeric probe JSONs where available: `cod reviewer/audits/<model>/03_numeric_probe.json`.
