# Baseline task-adaptation fixes (2026-11)

This pass implements the code-level findings of the second baseline audit.
It does not silently change official baselines: every behavioral change has a
CLI switch or an explicit deviation entry.  Model weights trained before the
changes marked **retrain** below must be retrained; prediction-only fixes can
be applied to existing checkpoints.

## Shared data/metric pipeline

| Fix | Files | Effect |
|---|---|---|
| Explicit dataset identity for test roots | `src/baseline_zoo/data.py`, `runtime.py` | Unseen/Novel stems such as `100_D.png`/`405.png` no longer become numeric dataset names. Support-based methods now receive the intended donor (CrackTree200 for crack-like sets, DRIVE for vessel-like sets) instead of always falling back to DRIVE. Support caches are also keyed correctly. |
| Literal 0/1 mask support | `src/baseline_zoo/data.py` | `load_mask` accepts both 0/255 and 0/1 masks; a 0/1 PNG can no longer silently become all-background. |
| Empty-mask metric convention | `src/core/metrics/binary.py`, `baseline_zoo/runtime.py`, `scripts/pipeline/metrics.py`, `collect_paper_metrics.py` | The default `penalty` policy scores an empty prediction as Dice/IoU/clDice=0 and HD95=image diagonal, so it cannot disappear from clDice/HD95 through NaN filtering. `--empty-metric-policy nan` restores the raw official semantics. |
| Partial-run filtering | `report.py`, `__main__.py` | The report skips runs whose `run_manifest.json` status is `partial`, `train_only`, `failed`, or invalid; use `--allow-partial` deliberately. Metric CSVs now carry the method group and dataset count. |
| Method grouping | `src/baseline_zoo/method_groups.py`, `report.py` | Markdown tables are separated into prompt-free supervised, proposed, support-based, oracle-prompted, semi-supervised, zero-shot, appendix and reconstruction groups. Oracle/support rows can no longer be mixed into the prompt-free main table. |
| Support cache reset | `matcher.py`, `rsrm.py`, `persam.py` | A reused adapter no longer reuses support tensors/embeddings from a previous prediction run. |
| Multi-group LR preservation | `runtime.py` | Official schedulers no longer overwrite per-group LRs (e.g. TPSegformer head x10, DKA head/adapter split); generic scheduler overrides still use the normalized base LR. |

## Confirmed model fixes

| Model | Fix | Retrain? |
|---|---|---|
| SCRWKV | Official `test.py` scales max-normalized logits to 0–255 and thresholds at 0.2; the equivalent normalized ratio is **0.2/255**, not 0.2. Exposed as `--scrwkv-threshold`. | No (prediction-only) |
| UCS | Default checkpoint selection is the shared 6-image validation Dice. `--ucs-selection train` restores the official training-Dice rule. | Yes if the selection change is used |
| SAM2-UNet | Default binarization is Otsu on the exported probability map, approximating the official adaptive threshold; `--sam2-unet-threshold fixed` restores 0.5. | No (prediction-only) |
| CWSAM | Binary loss now uses background 0.1 / foreground 1.0 CE plus 0.5 soft Dice instead of uniform CE. Weights are configurable; set all to neutral to restore the old port. | Yes |
| TPSegformer | Added a 0.5 soft-Dice term on the main TPHead output; the official Dice/edge terms only reached the auxiliary FCN head. | Yes |
| S2M-Net | Training crops are selected with the released foreground-filtered pool idea (up to 16 tries; minimum foreground 0.005). | Yes |
| SANSA | Prompt construction uses the number of supports actually returned by `select_supports`, avoiding `IndexError` on a short/empty donor set. | No |
| A2SAM | Stage-3 BCE/IoU/TA loss is computed at the model’s 1024 geometry instead of native 3504×2336 tensors. | Yes |
| nnU-Net 2D | A reused workdir clears old `results/Dataset502` and the model-directory pointer before staging/planning. | No |
| SegDINO v1 | `drop_last=False` for the 18-image protocol (the official loader had thousands of images and dropped 2/18 every epoch here). | Yes if reusing old runs |
| SegDINO v2 | Prediction no longer requires the original DINOv3 file; a random-init checkpoint requires explicit `--allow-random-init`. | No |
| SAM-Adapter | IoU loss has a zero-union epsilon, preventing NaN on empty/empty pairs. | Yes if a run hit NaN |
| C2P | Zero-shot metadata training no longer requires a protocol split manifest. | No |
| BLO-SAM | Selection defaults to shared validation Dice; `--blo-sam-selection d2` restores the official D2 criterion. | Yes if selection changes |
| SAM-OCTA | `--sam-octa-threshold` exposes the official 0.8 threshold for validation-only calibration; documentation now records the effective clDice weight (0.04 because the released loss already contains 0.8 Dice + 0.2 clDice). | No |
| MedCAGD | Warm-restart period scales the official T_0=5 over ~128 effective epochs to the configured budget (T_0≈31 at 800); `--medcagd-restart-epochs 5` restores the old trajectory. | Yes |
| MaskVSC | A stage-1-only fallback prints a warning and records `use_vig=false`; `--maskvsc-require-vig 1` fails instead of publishing a U-Net as MaskVSC. | No |
| CS-UNet | Batch-4 SGD LR is linearly scaled from the official batch-32 0.01 to 1.25e-3 (`paper_lr` keeps 0.01). | Yes |

## Task-adaptation geometry/selection

| Model | Fix | Retrain? |
|---|---|---|
| CS2Net | Default task adaptation: `--cs2net-geometry full` keeps native resolution capped at 1024 with /16 padding, `--cs2net-norm instance` replaces BatchNorm, batch 1, `--cs2net-augment mild`, and `--cs2net-threshold fixed`; the official 512 centre crop/BatchNorm/official augmentation/Otsu recipe is selectable with `center`/`batch`/`official`/`otsu`. The official 512 recipe underfits the mixed-domain 18-image pool (1000-epoch best 0.4185 Dice at E256; the 100-epoch check was 0.17); the adapted 100-epoch seed-0 check reached 0.5867 Dice / 0.6496 clDice / 25.69 HD95. | Yes (recipe and checkpoint schema changed) |
| BCU-Net | Default mask resize is NEAREST so 1-px labels are not erased by INTER_AREA; `--bcunet-mask-resize area` restores the official label path. Applied both to the offline augmentation dataset and the direct loader. | Yes |
| JTFN | Default `--jtfn-eval-mode tile` uses native-resolution overlapping 512-pixel tiles; `downscale` restores the old 1024 longest-side cap. | No (prediction-only) |
| LIOT | Default `--liot-eval-mode tile` uses 128×128 patches with averaging (stride 64 for 20-dataset throughput; `--liot-eval-stride 8` is release-exact). `whole` restores the old whole-image forward. | No (prediction-only) |
| FlowSDF | Default `--flowsdf-val-mode tile` uses native tiled SDF validation with stride 128 so checkpoint selection matches test geometry; `whole` keeps the fast 128 resize. | Retrain to change the selected checkpoint; existing predictions can be re-evaluated |
| KnowSAM | Default `--unlabeled-balance domain` round-robins the source domains instead of filling the 384-image pool with CORN/CRÉMI first; `source-order` restores the old policy. | Yes |
| SACM-STC fairness | `MATCH_BASELINE_GEOMETRY=1` on `eval_sacm_stc_20.sh` forces square resize and disables the empty guard, matching the baseline operating point for an ablation. | No |
| Resource budget | `run_config` now records `samples_per_epoch`, `batches_per_epoch`, `approx_optimizer_steps_per_epoch` and `approx_total_optimizer_steps`. | No |

## Remaining documented limitations

These are protocol/fidelity limitations, not silent bugs:

- SACM official preserves the upstream lack of SAM normalization, zero dense
  prompt, and stretched 60→800 schedule.  It is an official-code row, not a
  strong method baseline.  SACM-STC fixes correctness but also adds all-head
  supervision, deep supervision and IoU-head selection; it is not a pure bug
  fix.
- Support-based methods use own-domain train shots for Seen and a donor for
  Unseen/Novel, not the official same-class evaluation-fold supports.  Report
  them as a separate cross-domain support setting.
- Oracle methods (UCS, SAM-Med2D, SAM-OCTA) use test GT prompts and belong in
  a reference table.
- FOV masks are not applied; HD95 is in pixels; macro metrics weight datasets
  equally.  These are uniform protocol choices and must be stated.
- 800 epochs is not an equal optimizer-step budget; use the new metadata and
  run official-budget/step-matched ablations for claims about method ranking.
- Low-resolution whole-image methods (SegDINO/v2 256, BLO-SAM 256, SAM2-UNet
  352, ACO 352, C2P 384, TopoLoRA 384, SAMed/DKA/CS-UNet/TPSegformer/SCRWKV/
  K-Prism/BCU-Net/MaskVSC 512) remain resolution-limited.  This is an expected
  task-transfer limitation; a native-crop/tiled high-resolution variant is a
  separate experiment.
- A2SAM and DKA remain paper-level reconstructions of missing official code.
- The local shell is unavailable in this environment, so this pass was
  code-level; run `py_compile`/pytest in a shell-enabled session and smoke the
  changed adapters before the paper run.
