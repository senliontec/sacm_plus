# Other integrated models: static review (2026-10-26)

Scope: 15 models not covered by the seven-model post-A0 re-audit. Static code
review only; no training/GPU. P0 and selected P1 fixes were applied on 2026-10-26. A0 v3 is pass, but this review is
not a new per-model dual audit. Prior reports in `audits/<model>/README.md`
remain the detailed 01/02 evidence.

## P0 — fix before using these models

| model | finding | evidence | status |
|---|---|---|---|
| `fob_sam` | Structural donor selection is dead for every unseen/novel protocol root: `sample.dataset` comes from the unprefixed file stem (`data.py:48-50`), so names such as `111212-1` never match `CRACK_LIKE`/`VESSEL_LIKE`; all such datasets fall through to `DRIVE` (`fob_sam.py:284-306`). Crack-like unseen datasets receive vessel support. | `adapters/fob_sam.py:284-306`; `data.py:48-50`; agent CPU check: `DeepCrack -> DRIVE` | **fixed** `_canonical_dataset` |
| `sam_octa` | Final test prompts are unseeded: `prepare_eval_sample` calls `label_to_point_prompt_global` (`sam_octa.py:140`) which uses `random.sample` (`models/sam_octa/prompts.py:40-67`); prediction never seeds it. Same checkpoint + same `--seed` can produce different masks/metrics. | `adapters/sam_octa.py:140`; `runtime.py` prediction path; two fresh-process CPU probes differed | **fixed** eval prompt seeding |
| `fob_sam` | Official/pre-buffer checkpoint loading fails: the ignore set is `{'sin_pos'}` but the registered key is `masked_attention.sin_pos`; `load_state_dict` raises for the missing buffer. | `adapters/fob_sam.py:179-186`; `models/fob_sam/model.py:229-230`; official `3rd/FoB_SAM/models/FoB.py:217` | **fixed** ignore key |
| `fob_sam` | `_pair_cache` retains every test query image+mask for the whole prediction run: 3,580 protocol test images × 1 MiB ≈ 3.5 GiB CPU RAM; risky on the 11 GiB host and unnecessary because test masks are not used for prompts. | `adapters/fob_sam.py:99,216-224,421` | **fixed** query cache disabled |

## P1 — correctness/operational risks

| model | finding | evidence |
|---|---|---|
| `jtfn` | **fixed (downscale, not tiling)**: eval images longer than `--jtfn-eval-max-side` (default 1024) are resized before forward and upsampled to GT; disclosed memory adaptation. | `adapters/jtfn.py` eval path |
| `bcunet` | **fixed**: prediction / `--stage all` no longer requests pretrained downloads; training keeps the official initialization. | `adapters/bcunet.py` build_model |
| `samed` | **fixed**: low-res target is `input_size/4`. | `adapters/samed.py` |
| `samed` | **fixed**: poly horizon `max(epochs, 200)`, final epoch nonzero. | `adapters/samed.py` |
| `dscnet` | **fixed**: `--dsc-stride > --input-size` now raises instead of leaving uncovered stripes. | `adapters/dscnet.py` |
| `sam_octa` | **fixed**: `batch-size > 1` raises instead of dropping samples. | `adapters/sam_octa.py` |
| `sacm_official` | **fixed**: prediction builds with `checkpoint=None`; a host with only `best_model.pth` can evaluate. | `adapters/sacm_official.py` |
| `sacm_official` | **fixed**: `adapter_dim_ratio` and loss weights are saved/restored. | `adapters/sacm_official.py` |
| `sam_vit_l_unet` | **fixed**: checkpoint config stores SAM base path/SHA; a different encoder triggers a warning. | `adapters/sam_vit_l_unet.py` |
| `sam_vit_l_unet` | Port keeps image/mask fp32 and only quantizes features; reference cached image/mask as fp16. Logit difference was tiny (6.258e-7), but disclosure is partial. | `adapters/sam_vit_l_unet.py:85-101,119-120` |
| `cwsam` | **disclosed**: effective batch/workers difference added; class-weighted CE remains a disclosed high-risk adaptation. | `adapters/cwsam.py` |
| `sam_med2d` | **fixed**: train loss is averaged per batch. | `adapters/sam_med2d.py` |
| `fob_sam` | **fixed**: inference config and checkpoint paths are saved/restored. | `adapters/fob_sam.py` |
| `sam2_unet` / `segdino` / `fob_sam` | **disclosed**: official seed/workers differences added to deviations. | adapters' `deviations` |

## P2 — low-risk or reporting-only

- `jtfn`: prediction uses strict `>` while official uses `>= 0.5` (`adapters/jtfn.py:195-199` vs `3rd/JTFN/test.py:88-89`); only exact ties differ.
- Shared runtime logs the LR after `scheduler.step()` (`runtime.py:178-179,617-635`), so CS2Net/JTFN history shows the post-step LR for the epoch just trained. Training is unaffected.
- `topolora`: adapter default `--grad-clip 0.0` contradicts the disclosed 1.0 recipe; the model YAML passes 1.0, so only direct CLI use is affected.
- `sam_octa`: clDice applied to all binary datasets; official applies it only to Artery/Vein/LargeVessel (disclosed).
- `samed`: AdamW vs source-default SGD (disclosed); random rotation after resize to 512 differs from official native-resolution augmentation (undisclosed).
- `cwsam`: final metrics are computed after resizing probabilities to native GT rather than official 1024-resolution argmax; magnitude unquantified.
- `segdino`: official DINOv3 backbone/hub path absent; same-state parity remains unverified.

## Post-fix GPU/CPU smoke (2026-10-26)

- **JTFN high-resolution memory** (GPU 0, random-init model, 1024 max side):
  - HRF 3504×2336 → model 1024×683, padded 1024×704, peak allocated 3.72 GiB, reserved 4.49 GiB.
  - FIVES 2048×2048 → model 1024×1024, peak allocated 5.33 GiB, reserved 6.22 GiB.
  - Both fit the 10.8 GiB free GPU; predictions restored to native shape.
- **FoB donor/cache** (CPU, DeepCrack sample `111212-1`): canonical dataset `DeepCrack`, donor `CrackTree200`; query not in `_pair_cache`, one support entry cached.
- **SAM-OCTA seed** (two fresh CPU processes, same checkpoint-less prompt path): all 26 prompt coordinates and labels identical with `--seed 7`.

These are smoke checks only; they do not replace G6a/G6c or a full model run.

## Suggested fix order

1. `fob_sam`: donor-name resolution, `sin_pos` ignore, clear/limit `_pair_cache`.
2. `sam_octa`: seed the test prompt draw and guard batch-size > 1.
3. `jtfn`: choose resize/tiling for full-resolution eval, then test on one high-resolution sample.
4. `bcunet`: do not download pretrained weights when a checkpoint is supplied or prediction only.
5. `samed`: derive low-res target from input size; fix poly horizon/last-step zero.
6. `sacm_official` / `sam_vit_l_unet`: persist construction/provenance config.
7. Disclosure-only items: CWSAM, SAM2-UNet, SegDINO, FoB workers/seed, DSCNet stat wording.

P0 and the selected P1 fixes above were applied on 2026-10-26; remaining items are unchanged. See `POST_FIX.md`.
