# Baseline fidelity classification

This table separates **paper placement** from **current gate status**. A model in
`main-adapted` is not main-table eligible until A0/G0–G5/G6a/G6c pass; the
column only says what the method is, and where it can honestly be presented.

Legend:

- `architecture`: `exact` = official model code/config (task head may be
  configured for binary output); `backbone-changed` = different encoder;
  `repaired` = official code was unrunnable and had to be repaired;
  `extension` = project-only architecture.
- `recipe`: `official` = official training recipe preserved; `adapted` =
  materially changed for the unified 3-shot/20-dataset protocol;
  `training-free` = no method training.
- `inference`: `official` = official inference path; `adapted` = GT export or
  shared evaluator; `prompt-free` = empty/no-mask prompts; `oracle` = GT-derived
  prompts at test time; `support` = support/reference images;
  `semi-supervised` = unlabeled pool is part of the method.
- `weights`: `retrained` = trained from official backbone/scratch;
  `missing` = required official encoder/backbone is absent; `base-only` =
  only a SAM/base checkpoint is used.

## Classification

| model | architecture | recipe | inference | weights | proposed placement | note |
|---|---|---|---|---|---|---|
| `unet` | exact code, binary head | adapted | official | retrained | **main-adapted** | official architecture/loss/scale preprocessing/logit upsample; 3-shot recipe adapted (scale=1.0, batch 1, AdamW 1e-3/cosine, flips/rot90) because the literal official RMSprop/ReduceLROnPlateau recipe collapsed LR on 18 images; disclose as "3-shot U-Net" |
| `cs2net` | exact | official | adapted | retrained | **main-adapted** | official train recipe; official predict.py is unrunnable, port rewrites center-crop/Otsu/GT placement |
| `bcunet` | repaired | approximate | adapted | retrained | **appendix** | official code has SyntaxError/undefined upconv2x2; loss/fused semantics changed; not an official reproduction |
| `maskvsc` | exact | official | adapted | retrained | **main-adapted** | protocol/seed and GT export adapted; architecture and loss preserved |
| `nnwnet` | exact + official trainer | official | official | retrained | **main-adapted** | official nnU-Net plugin/trainer, 18/6 protocol replaces 5-fold CV |
| `jtfn` | exact | adapted | adapted | retrained | **main-adapted** | StepLR/augmentation approximated; eval >1024 is downscaled for 11 GiB memory |
| `dscnet` | exact | official | adapted | retrained | **main-adapted** | official recipe; PIL/patch inference replaces MONAI/SimpleITK; GT export |
| `segdino` | backbone-changed | official | adapted | missing | **appendix** | timm DINOv3 instead of the official hub/backbone; no same-state parity |
| `sam_med2d` | backbone/init-changed | adapted | oracle | retrained | **reference** | adapter code exact, but ViT-L vs the official ViT-B checkpoint; SAM decoder init, not SAM-Med2D; GT-box prompted; not prompt-free |
| `sam_octa` | backbone/loss-changed | adapted | oracle | retrained | **reference** | ViT-L vs official ViT-H; binary clDice; GT point prompts; not prompt-free |
| `cwsam` | backbone/head/loss-changed | adapted | prompt-free | retrained | **supplement** | ViT-L vs official ViT-B, 2-class head, neutral CE; disclose |
| `sam_adapter` | exact | official | prompt-free | retrained | **main-adapted** | official 20-epoch recipe; author weights absent, retrained from SAM ViT-L |
| `topolora` | exact ViT-B | official | prompt-free | retrained | **main-adapted** | official 50-epoch recipe; partial accumulation/PT runtime differences disclosed |
| `samed` | head-changed | adapted | prompt-free | retrained | **supplement** | official 8-class head/weights replaced by binary head; AdamW source ambiguity |
| `fob_sam` | exact FewShotSeg + SAM, 2D adaptation | adapted | support | retrained | **reference** | official support is test-fold held-out; port training/donor support; G5 still fail |
| `sacm_official` | exact | official | prompt-free | retrained | **main-adapted** | official 60-epoch recipe; sigmoid threshold fix and 20-dataset protocol disclosed |
| `persam` | exact training-free | training-free | support (one-shot) | base-only | **reference** | 3-shot majority-vote extension; not main prompt-free |
| `matcher` | exact training-free | training-free | support | missing | **reference** | DINOv2 ViT-L/14 absent; FSS score/seed fixed; cannot claim official reproduction |
| `knowsam` | exact, 2-class head | adapted | semi-supervised | retrained | **supplement** | official ACDC is 4-class; port uses a 2-class head. 3+3 batch and unlabeled pool only CHASEDB1/CREMI/CORN; test exclusion fixed |
| `sam2_unet` | exact (Hiera-L missing) | adapted | adapted | missing | **probe/appendix** | official SAM2-UNet architecture, not an extension; SAM2 Hiera-L checkpoint absent; 20-epoch/batch-12 recipe replaced by 200-epoch/batch-4 protocol |
| `sam_vit_l_unet` | extension | adapted | adapted | base-only | **probe** | project SAM+U-Net probe, not a paper baseline |
| `ucs` | exact HEAD, paper drift | official | oracle | retrained | **reference** | oracle GT prompts; HEAD differs from paper modules; no official weights |

## Confirmed task difference (2026-10-26)

The official task/data/prompt protocol is not the same as our 20-domain,
18-train/6-val, 3-shot binary protocol for any of the 22 methods. The model
architecture can often be reused, but the original task cannot be copied
wholesale. The differences fall into three groups.

### A. Same binary dense-segmentation type, different corpus and budget (13)

| model | official task/data | port difference |
|---|---|---|
| `unet` | Carvana RGB car masks; official 2-class output (`3rd/Pytorch-UNet/train.py:182`) | 20 mixed domains; 1-class head; 3-shot adapted AdamW/cosine/scale=1.0; 800-epoch budget |
| `cs2net` | DRIVE/STARE retinal vessels (`3rd/CS-Net/README.md`) | 20 domains; official train recipe; inference rewritten |
| `bcunet` | DRIVE/DR HAGIS/KVASIR/CVC binary (`3rd/BCU-Net/README.md:43`) | 20 domains; official code repaired |
| `maskvsc` | five retinal vascular datasets, binary | 20 domains; architecture/loss preserved; selection/export adapted |
| `nnwnet` | nnU-Net biomedical benchmark, 2D/3D, 5-fold CV | 2D, 20 domains, 18/6 3-shot; official trainer |
| `jtfn` | DRIVE/Crack500/Roads/CrackTree200 binary | 20 domains; simplified augmentation; eval downscale |
| `dscnet` | DRIVE/roads 2D, KIPA 3D tubular structures | 20 domains; official recipe; inference adapted |
| `segdino` | TN3K/Kvasir/ISIC binary | 20 domains; timm DINOv3 instead of official hub backbone |
| `sam_adapter` | COD/polyp binary | 20 domains; official 20-epoch recipe, ViT-L |
| `topolora` | DRIVE/STARE/CHASE/Kvasir/SAR binary (`3rd/Seglab/README.md`) | 20 domains; official ViT-B/50-epoch recipe |
| `sacm_official` | four internal curvilinear datasets (`3rd/SACM/README.md:29`) | 20 domains; official 60-epoch recipe |
| `sam2_unet` | COD/SOD/MAS/mirror/polyp binary | 20 domains; Hiera-L absent; 200 epochs/batch 4 |
| `sam_vit_l_unet` | project extension, no official task | 20 domains; project probe |

### B. Output head or supervision semantics changed (3)

| model | official task | port difference |
|---|---|---|
| `cwsam` | SAR 5/10-class semantic segmentation, ViT-B | 2-class binary head, ViT-L, neutral CE |
| `samed` | Synapse 8-class organ segmentation | binary 1-class head; retrained |
| `knowsam` | ACDC 4-class (Endoscope/BCSS 2-class), 12 labeled + 12 unlabeled | 2-class head; 3+3 batch; pool limited to CHASEDB1/CREMI/CORN |

### C. Prompt or support semantics differ (6)

| model | official task/inference | port difference |
|---|---|---|
| `sam_med2d` | 4.6M medical images, box/point/mask prompts; official ViT-B `sam-med2d_b.pth` | GT boxes; ViT-L; SAM decoder init; retrained on 18 images |
| `sam_octa` | OCTA-500, point prompts, ViT-H | GT points; ViT-L; binary clDice |
| `ucs` | curvilinear segmentation with GT boxes/points at test (`3rd/UCS/test.py`) | same oracle-prompt semantics; 20-domain 3-shot |
| `fob_sam` | 3D FSMIS (abdominal CT/MRI, skin), support from eval fold, 1-shot | 2D 3-shot; training/donor support; G5 fail |
| `persam` | one-shot personalized segmentation, training-free, ViT-H | 3-shot majority vote; training-free |
| `matcher` | one-shot semantic/part segmentation, DINOv2 + SAM, training-free | 3-shot; DINOv2 ViT-L/14 absent |

This confirms the distinction used by the table:

- **Architecture** can be transferred where the official code/weights exist;
  `exact`, `backbone-changed`, `repaired`, and `extension` describe that
  dimension only.
- **Task, data, prompt source, support policy and training budget** cannot be
  transferred unchanged. A binary-output model still cannot be trained or
  evaluated on the official corpus without changing the paper's protocol.
- Therefore an official architecture is necessary but not sufficient for an
  official-number claim. Only a same-task, same-recipe, same-checkpoint run
  could support that claim, and none of the 22 ports is such a run.

## Counts

- **main-adapted candidates (9)**: `unet`, `cs2net`, `maskvsc`, `nnwnet`,
  `jtfn`, `dscnet`, `sam_adapter`, `topolora`, `sacm_official`
- **supplement (3)**: `cwsam`, `samed`, `knowsam`
- **reference (6)**: `sam_med2d`, `sam_octa`, `fob_sam`, `persam`, `matcher`,
  `ucs`
- **appendix/probe (4)**: `bcunet`, `segdino`, `sam2_unet`, `sam_vit_l_unet`

`main-adapted` means “method is appropriate for the main comparison, but the
number is ours under a unified protocol; the paper must disclose the adaptation
and must not quote the original paper’s number as ours.”

## Decisions still required

1. Whether `main-adapted` models are acceptable in the main table with a
   footnote, or require a separate adapted table.
2. Whether `cwsam`/`samed` binary-head adaptations belong in supplement or main
   once their G6c runs exist.
3. Whether `sam_med2d`/`sam_octa` (oracle) and `fob_sam` (support protocol fail)
   should be reference-only; they cannot sit in a prompt-free main column.
4. Whether `bcunet`/`segdino` are worth keeping as appendix baselines at all.

This is a paper-scope decision, not a request to change the code. No model is
main-table eligible until its gates, especially G6c, pass.
