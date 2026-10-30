# 40-baseline porting audit

Date: 2026-10-28 (fix pass updated 2026-10-29)
Scope: all 40 runnable official/reference baselines in `configs/models/`.
Method: read-only comparison of the local paper (`parsing/`), official code
(`3rd/`), port (`src/baseline_zoo`), config, `UPSTREAM.md`, and tests.
No training or GPU jobs were run for this audit.

Severity: **P0** unusable/invalid result; **P1** material fidelity or
comparability issue; **P2** minor/latent/documentation issue; **P3**
provenance/license issue. `PASS` means no confirmed porting bug; documented
deviations may remain. `WARN` means a real but non-blocking issue.

## 1. Consolidated table

### 1.1 Classic / CNN group

| model | paper / official code | verdict | severity | main issue | status |
|---|---|---|---|---|---|
| `unet` | U-Net 2015; local paper absent; `3rd/Pytorch-UNet` is third-party milesial | WARN | P2 | architecture/BCE+Dice faithful, but optimizer/input/class/augmentation recipe is non-official | documented; label as 3-shot U-Net |
| `bcunet` | BCU-Net CBM 2023; paper now local; `3rd/BCU-Net` unrunnable (NameError, missing warmlearnrate) | WARN | P1/P2 | paper main method is MRL, average fusion 0.5/0.5, bilinear U-Net, 18× augmentation, 100 epochs; release main.py uses CE, 0.4/0.6, ConvTranspose, no augmentation | **paper profile implemented**: RecallCE on ConvNeXt/U-Net/fused, 0.5/0.5 fusion, bilinear Up repaired, 18-variant offline augmentation (amplitudes reconstructed), 100 epochs/batch 2; CE/0.4-0.6/no-aug remain ablations; OutConv 1×1 logits reconstruction |
| `cs2net` | **CS2Net-2D**; official `3rd/CS-Net/model/csnet.py` + `train.py` (iMED-Lab) | PASS | P1 | 2D vs 3D identity confusion; `model/csnet_3d.py` is a separate Conv3d extension | **clarified**: 2D branch is correct for DRIVE/CHASE/DCA1/CORN; 3D extension cannot consume 2D protocol images without a private depth-1 adaptation; Otsu verified |
| `csunet` | CS-UNet ICASSP 2026; `3rd/CSUNet/CSUNet.ipynb` only | WARN | P2 | notebook DynamicSnake is a placeholder; paper PACB BN/Snake differs; no weights/license | label released-code variant; document |
| `dconnnet` | DconnNet CVPR 2023; paper absent | PASS | P2 | faithful; HSV hue wrap and no LICENSE; ResNet-34 present | document |
| `dscnet` | DSCNet ICCV 2023; `3rd/DSCNet` MIT | WARN | P1 | paper TCLoss is absent from official code and port; port adds Dice; 800-epoch deviation now explicit | **doc fixed**: “no TCLoss” deviation added; implementing TCLoss would be a non-official extension |
| `jtfn` | JTFN ICCV 2021 | WARN | P2 | official FOV `ignore_mask` omitted from evaluation | **doc fixed**: shared protocol applies no FOV mask; deviation now explicit |
| `liot` | LIOT TIP 2022 | FAIL→PASS | P1 | bright-polarity table inverted dark retinal datasets (DRIVE/CHASE/etc.) | **fixed**: retinal dark; bright set CORN/DeepGlobe/ROAD/TopoMortar/WIRE |

### 1.2 Remaining CNN / flow / nnU-Net group

| model | paper / official code | verdict | severity | main issue | status |
|---|---|---|---|---|---|
| `flowsdf` | FlowSDF IJCV 2025 | PASS | P2 | K=4 test averaging is released config, not paper K=25; SDF notebook not vendored | **doc fixed**: both deviation entries now state released K=4 vs paper K=25 |
| `maskvsc` | MaskVSC TMI 2025; paper in `parsing/` | PASS | P3 | license contradictory (Apache vs all-rights-reserved); paper L_C unstable on sparse domains | **release main restored**: official patch16 ViG decode/soft labels; release L_C/flip-only/dot-product KNN default; paper L_C/rotation/Euclidean as ablations; 100-epoch seed-0 stage-1 A/B 0.4612/0.4726 (release, rot0/30) vs 0.1133/0.1636 (paper); paper L_C 10-100× BCE on sparse masks and needs `--maskvsc-connect-cap`; pooled 512/600+200; resolve license |
| `medcagd` | MedCAGD ECCV 2026 | PASS | P2 | soft Dice added; 256/128 patches for all domains; batch adaptation | documented |
| `nnunet2d` | nnU-Net Nature Methods 2021; paper absent; runtime uses installed nnunetv2 | PASS | P3 | runtime paths statically checked only; train-only planning | install/smoke on server |
| `nnwnet` | nnWNet CVPR 2025; paper in `parsing/` | WARN | P2 | paper nnU-Net 2.3.1/200/5-fold/ASD vs project 2.8.1/800/18-6/shared metrics; no license | **protocol + monitoring unified**: shared run_config/val schema, full-image val + val-Dice selection every epoch (interval default 1), patch pseudo-Dice under `nnunet/*`; WNet 7.04M; train-only planning; stale outputs cleared; add license |
| `s2mnet` | S2M-Net ICML 2026 | PASS | P2 | cosine had no 1e-6 floor; stage text says spectral-only 1–2 (code: spectral 1–5, gated 3–5) | **fixed** LR floor; stage description verified accurate |
| `scrwkv` | SCRWKV ICML 2026 | FAIL→PASS | P0 | bool 0/1 mask cubic-resized then thresholded at 127 → all-empty targets | **fixed**: 0/255 before resize; added foreground assertion; old results invalid |
| `tpsegformer` | TPSegformer CVPR 2026 | PASS | P1 | no Swin-T ImageNet checkpoint, defaults random init; Hilbert path is a no-op upstream | **fixed/verified**: official OpenMMLab Swin-T loads with MMSeg→timm key mapping (missing=0); warmup starts at 1e-6 |

### 1.3 SAM PEFT group

| model | paper / official code | verdict | severity | main issue | status |
|---|---|---|---|---|---|
| `convlora` | Conv-LoRA ICLR 2024; AutoGluon | PASS | P2 | documented 800 vs 30/100, decoder training, flip p=0.5, Adam vs AdamW | keep documented |
| `blo_sam` | BLO-SAM ICML 2024 | PASS | P2 | documented 9/9 D1/D2 split and all-D2 update deviations | keep documented |
| `c2p` | C2P ECCV 2026 | PASS | — | zero-shot, frozen official checkpoint; training-free path handled | none |
| `sam_adapter` | SAM-Adapter ICCV 2023 | PASS | P2 | YAML said only prompt generator+decoder train; prompt encoder pe_layer/no_mask_embed also train | **fixed** YAML note |
| `samed` | SAMed arXiv 2023, often mislabeled MICCAI | WARN | P2 | venue error; large documented adaptation incl. high-res head | **fixed** config/adapter/model docstrings |
| `topolora` | TopoLoRA-SAM ICPR 2026 | PASS | P2 | documented 800 vs 50, fp32 vs AMP | keep documented |
| `sam2_unet` | SAM2-UNet arXiv 2024 | WARN | P2 | `--random-init` ignored when checkpoint exists | **fixed** + test |
| `sansa` | SANSA NeurIPS 2025 | FAIL→PASS | P0 | real model returns 3-D logits; adapter kept 3-D and interpolate crashed | **fixed** 4-D query slice + real-contract test; preprocessing doc corrected (official 518/640→1024 vs port direct 1024) |

### 1.4 Support / prompt / oracle group

| model | paper / official code | verdict | severity | main issue | status |
|---|---|---|---|---|---|
| `fssam` | FSSAM ICML 2025 | PASS | P1 | default DINOv2-L, paper main is DINOv2-B; doc said bicubic, official/code use bilinear | **fixed**: default official SAM 2-S + DINOv2-B; DINOv2-L remains an ablation; bilinear doc corrected |
| `matcher` | Matcher ICLR 2024 | PASS | P2 | training-free; only documented 518/GT export/support choices | none |
| `persam` | PerSAM ICLR 2024 | PASS | P2 | one-shot official → 3-shot vote; support failures could be silently voted empty | **fixed**: failures now raise with support/dataset context; no empty-mask fallback |
| `rsrm` | RSRM ECCV 2026 | PASS | P1 | local DINOv3 ViT-S vs paper ViT-B; sklearn vs fast k-means | **fixed**: default official DINOv3 ViT-B/16; ViT-S remains an explicit ablation |
| `sam_med2d` | SAM-Med2D; paper absent | WARN | P1 | official checkpoint released as ViT-B (`sam-med2d_b.pth`); oracle GT boxes | **fixed and verified**: official ViT-B loaded (2.4 GB, PyTorch 2.6 `weights_only=False`), 256 forward/prediction smoke passed; oracle labeling remains |
| `sam_octa` | SAM-OCTA; paper absent | WARN | P1 | ViT-L vs official ViT-H; RGB instead of three OCTA projections; oracle points | **configurable** ViT-B/L/H + matching checkpoint; L default for 24 GiB; oracle labeling remains |
| `cwsam` | CWSAM; paper absent | PASS | P1 | released configs are ViT-B; paper's shared-ViT-L claim unverified | **fixed**: default is ViT-B + official B checkpoint; ViT-L exposed as `--cwsam-encoder vit_l`; binary CE documented |
| `ucs` | UCS ICCV 2023 | WARN | P1 | oracle GT prompts while paper claims prompt-free; release HEAD differs; init_mask range was 1..7 vs official 1..6 | **range fixed** (`randrange` half-open); oracle labeling/paper modules remain a protocol decision |

### 1.5 A2SAM / SAM variants group

| model | paper / official code | verdict | severity | main issue | status |
|---|---|---|---|---|---|
| `a2sam` | A2SAM MIA 2026; `3rd/A2SAM` partial | WARN | P1 | release lacks U-Net/prior/DMT/SPG/weights; port reconstructs them; no end-to-end parity | label paper-level reconstruction; P0/P1 fixes below |
| `aco_moe_lora` | ACO-MoE-LoRA ICML 2026 | WARN | P2 | ACO scheduler stepped per 4-image micro-batch vs once per batch-12; cooldown ~3x faster; license missing | **documented** accumulation artifact; cadence left unchanged to avoid untested behavior |
| `knowsam` | KnowSAM; paper absent | WARN | P1 | batch 6 vs official 24; unlabeled pool only CHASEDB1/CREMI/CORN; no test leak | match effective batch/pool or document |
| `sacm_official` | SACM CVPR 2026 | WARN | P2 | 60→800, val Dice vs official F1/loss, sigmoid correction of upstream raw-logit threshold; license missing | document; add license |
| `segdino` | SegDINO; paper absent | PASS | P2 | README says frozen backbone, but official trainer never calls lock_backbone and port trained all | **fixed**: frozen backbone is default; `--segdino-train-backbone` reproduces released full-finetune |
| `segdino_v2` | SegDINO v2; paper absent | WARN | P2 | hybrid paper-frozen + released BCE/augmentation; weight decay differs | declare one profile; add paper |
| `sam_vit_l_unet` | project extension, no paper/3rd | PASS (N/A) | P2 | Dice epsilon 1.0; feature-level flip/rot after frozen ViT; square stretch | **eps fixed** to 1e-6 + test; still exclude from official table |
| `dka` | DKA ICLR 2026; official segmentation code absent | WARN | P2 | Segmenter-B host/loss/augmentation reconstructed; release has no segmentation entrypoint | **provenance verified**: UPSTREAM labels the reconstruction and active DKA path |

## 2. P0 confirmed and fixed

1. **SCRWKV all-background training targets** (`adapters/scrwkv.py`):
   `load_mask` returns bool; cubic resize of 0/1 followed by `>127` gives an
   all-zero mask. Fixed by converting to 0/255 before resize; added a
   non-empty-foreground test. All old SCRWKV results are invalid.
2. **SANSA validation/prediction crash** (`adapters/sansa.py`):
   real model returns `(B*T,H,W)`; `logits[-1:]` stays 3-D and
   `F.interpolate` raises. Fixed with `logits[-1][None,None]`; fake test now
   returns the real 3-D contract.
3. **A2SAM checkpoint/LR/threshold hazards** (already fixed in the current
   tree): E201 cannot become best, stage-specific LR logging,
   `coarse_threshold` wiring, stage snapshots. Branch-cap/DCA1 triage remains
   open.

## 2b. Confirmed P1/P2 issues fixed mechanically

- **LIOT retinal polarity**: DRIVE/CHASEDB1 and other retinal fundus sets are
  dark-vessel images and must not be inverted; only CORN/DeepGlobe/ROAD/
  TopoMortar/WIRE are bright-foreground. Tests updated.
- **SAM2-UNet `--random-init`**: the flag now really bypasses an existing
  checkpoint; regression test added.
- **S2MNet cosine floor**: LR now reaches the official 1e-6 floor instead of
  zero.
- **SAMed venue**: corrected from MICCAI 2023 to the arXiv 2023 technical
  report (often cited as MICCAI).
- **SAM-Adapter config**: now states that the prompt encoder's
  `pe_layer`/`no_mask_embed` are also trained.
- **CS-Net identity**: `cs2net` is now labeled CS-Net (MICCAI 2019,
  iMED-Lab), not CS2Net (MedIA 2020); the legacy id is retained.
- **UCS init-mask sampling**: `randrange(1, iter_point-1)` now matches the
  official half-open `np.random.randint(1, iter_point-1)` range.
- **TPSegformer warmup**: starts at 1e-6 of the base LR as in PolyLrUpdater;
  regression test added.
- **FlowSDF K samples**: docs now consistently state released K=4 vs paper
  K=25.
- **JTFN FOV**: the omitted official `ignore_mask` is now an explicit
  shared-protocol deviation.
- **DSCNet TCLoss**: the absent paper loss is now an explicit deviation.
- **nnWNet docs**: planning is train-only; validation cases are staged only
  after planning.
- **SANSA preprocessing**: official 518/640 intermediate resize vs direct
  1024 in the port is documented.
- **FSSAM DINO resize**: docs corrected from bicubic to the release's
  bilinear 560 resize.
- **ACO scheduler cadence**: 4-image micro-batch stepping vs 12-image
  batches is documented.
- **CWSAM backbone**: default switched to the released ViT-B/checkpoint;
  ViT-L remains an explicit ablation.
- **SAM-Med2D**: default official ViT-B `sam-med2d_b.pth` loads with
  PyTorch 2.6 (`weights_only=False`), 256-input prediction smoke passes;
  decoder input size is configurable.
- **SAM-OCTA backbone**: ViT-B/L/H and matching checkpoint selection are
  configurable; ViT-L remains the 24 GiB default.
- **SegDINO freeze**: default now matches the README/paper frozen backbone;
  released full-finetune is an explicit flag.
- **PerSAM failures**: support errors now fail fast instead of silently
  contributing an empty vote.
- **FSSAM**: default official SAM 2-S + DINOv2-B, verified real forward;
  DINOv2-L remains an ablation.
- **RSRM**: default official DINOv3 ViT-B/16, verified real forward; ViT-S
  remains an ablation.
- **TPSegformer**: official OpenMMLab Swin-T loads with MMSeg→timm key
  mapping (missing=0) and real forward passes.
- **SAM ViT-L + U-Net**: Dice epsilon corrected to 1e-6.

## 3. P1 confirmed but not silently changed

These need a method/paper decision rather than a mechanical patch:

- DSCNet lacks the paper's TCLoss in both official 2D code and the port;
  the port is now explicitly labeled “no TCLoss”.
- SAM-Med2D/SAM-OCTA remain oracle-prompted adaptations; their input-size
  and backbone options are fixed, but the protocol labels remain.
- UCS is oracle-prompted despite the paper's prompt-free claim; the
  init-mask range is fixed, labeling remains.
- KnowSAM's effective batch (6 vs 24) and unlabeled pool differ materially.
- A2SAM is a paper-level reconstruction, not an official end-to-end model.
- The row is the 2D CS2Net/CS-Net implementation; a true 3D CS2Net
  (`csnet_3d.py`) would be a separate volume baseline, not a repair of this
  2D row.

## 4. Cross-cutting findings

- **Reporting key mismatch fixed**: `visualize_predictions.py` used
  `sam-med2d`/`sam-octa`/`sacm` instead of `sam_med2d`/`sam_octa`/
  `sacm_official`, and lacked `sam2_unet`; labels now cover every registered
  baseline config.


- **Epoch budgets are not uniform**: BLO-SAM=100; CS-Net/JTFN=1000;
  KnowSAM=50000 iterations; c2p/matcher/persam/rsrm=0 (training-free);
  others mostly 800. The one-click 40-baseline run uses each config default,
  not a global 800.
- **5 models still have no dedicated end-to-end test file**: cs2net,
  knowsam, nnwnet (only incidental runtime tests), sacm_official,
  sam_med2d (only an init test).
- **Mixed protocol categories** must be separated in paper tables:
  prompt-free, support-based, oracle-prompted, semi-supervised,
  training-free.
- **Missing local papers**: U-Net, CS2-Net, DconnNet, nnU-Net, KnowSAM,
  SegDINO, SegDINO v2, SAM-Med2D, SAM-OCTA, CWSAM.
- **Missing/partial official implementations or weights**: A2SAM (partial),
  BCU-Net (broken), DKA (no segmentation); SAM-Med2D now uses the official
  ViT-B checkpoint but remains oracle-prompted. Missing licenses: ACO,
  KnowSAM, SACM, BCU-Net, CS-UNet, DconnNet, JTFN, MaskVSC conflict.
- Local `baseline_zoo doctor`: 0 required assets/dependencies missing and
  0 optional main-table assets unavailable (Swin-T, DINOv2-B, DINOv3-B,
  `sam-med2d_b.pth` all present).

## 5. Verification after fixes

- Targeted tests pass for SANSA, SCRWKV, LIOT, SAM2-UNet, S2MNet,
  TPSegformer, CWSAM, SegDINO, SAM-Med2D, SAM-OCTA, PerSAM and the SAM
  ViT-L+U-Net extension. Official Swin-T (Swin→timm key mapping), SAM 2-S +
  DINOv2-B, DINOv3 ViT-B/16 and SAM-Med2D ViT-B all load and pass their
  forward/prediction smokes. The full pytest suite passes (exit 0),
  ruff passes on all changed Python files, and the 40-model launcher
  dry-run emits exactly 40 commands with exit 0.
- SCRWKV and LIOT checkpoints/results produced before this audit are
  invalid and must be retrained. Any UCS run from the old inclusive index
  range is also invalid. Models whose preprocessing/backbone labeling
  changed only in docs do not need retraining.
- No GPU training was run for this audit; no metric in this report is a
  result claim.

## 6. Remaining blockers before the 8×3090 seed-0 run

- All four main-table assets are present and verified: Swin-T,
  DINOv2-B, DINOv3-B and `sam-med2d_b.pth`. No further weight download is
  required for TPSegformer/FSSAM/RSRM/SAM-Med2D.
- Decide whether to add a true CS2Net (MedIA 2020) baseline; the current
  `cs2net` row is CS-Net.
- Resolve missing licenses (ACO, KnowSAM, SACM, BCU-Net, CS-UNet,
  DconnNet, JTFN, MaskVSC conflict) before public release.
- A2SAM branch-cap/DCA1 triage still needs the server 3090 stage-3 run.

