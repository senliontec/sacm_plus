# MaskVSC

- Upstream: https://github.com/Zhouyi-Zura/MaskVSC
- Vendored snapshot: `3rd/MaskVSC`, HEAD
  `5e60565b5ec63cb91295e06b9b502d6350e89d76`.
- Paper: `parsing/[2025 tmi] Masked Vascular Structure Segmentation and
  Completion in Retinal Images` (Zhou et al., IEEE TMI 2025).
- License: the tree contains `LICENSE.txt` (Apache-2.0), but the README says
  "All rights reserved. The code and data are released for academic research
  use only." This conflict must be resolved before redistribution.
- Files copied:
  - `model/UNet.py` -> `unet.py` (unchanged)
  - `lib/Loss.py` -> `loss.py` (release arithmetic unchanged; device-safe
    kernel and a `paper`/`release` mode selector added)
  - `ImageDataset_Fundus` corruption branch -> `transforms.py`
    (extracted verbatim, plus an empty-skeleton guard)
  - `model/gcn.py` -> `gcn.py` (ViG completion head; layers unchanged,
    `decode` and `graph_metric` selectors added)
- Added wrapper: `full.py` (`MaskVSCFull`, stage-1 UNet + stage-2 ViG)
- Runtime consumer: `baseline_zoo.adapters.maskvsc.MaskVSCAdapter`

## Paper-code conflict ledger

The paper and the released code disagree in several places. The **main
baseline follows the released code**; paper variants remain explicit
ablations because the paper-mode L_C reconstruction is unstable on the
pooled sparse 3-shot data.

| Topic | Paper | Release code | Main default / ablation |
|---|---|---|---|
| Connectivity loss | Eq.1 divides by foreground pixels; Eq.2 S is a mean over N; no morphological opening mentioned | `count_nonzero(lab_after_sigmoid)` counts every pixel, after a 3x3 opening | **release**; paper mode `|S(Y)-S(Y_G)|*N/#foreground` skips opening and supports `--maskvsc-connect-cap` (unstable unbounded; recalibrate) |
| Stage-1 augmentation | random rotation ±30° + horizontal flip | horizontal flip only; no rotation anywhere | **release flip-only**; paper rotation via `--maskvsc-rotation-degrees 30` |
| Stage-2 augmentation | not separately specified | none in `ImageDataset_GCN_*` | none (release) |
| ViG KNN | Euclidean distance | dot-product top-k | **release dot-product**; paper Euclidean via `--maskvsc-vig-graph paper` |
| Stage-1 freeze in stage 2 | explicitly frozen | not `eval()`; only absent from the optimizer, so BatchNorm buffers drift | paper frozen + `eval()` |
| Input size | per dataset: 512 DRIVE/RITE, 576 STARE, 1024 LES-AV/2PFM, 384/304 OCTA-500, 304 ROSE | Fundus 512, OCTA 384, 2PFM native | unified 512 benchmark input |
| Corruption | Fundus SD20/7×7, OCTA SD100/3×3, 2PFM SD20/11×11 | matches per-modality classes | unified SD20/7×7 pooled protocol |
| Epochs | 200 per training step (200+200) | 200+200 | 600+200 under the shared 800-epoch budget |

The printed L_C equations are scale-degenerate: Eq.2 normalizes S by N while
Eq.1 divides by foreground pixels, so the literal combination is ~1e-6 and
numerically inert. Paper mode restores the compensating N factor; on a real
DRIVE 512 mask with an imperfect prediction it gives BCE 0.307, release
L_C 1.5e-7 and paper L_C 0.665. However, the correction is not uniformly
valid across modalities. Random-prediction measurements per 512 mask:
DRIVE/CHASE 2.5-4.8, CORN 12-33, CREMI 28-37, CrackTree200 62-166, while
BCE ~1.0. With λ=1, sparse domains dominate mixed batches and the stage-1
model collapses on CORN/CrackTree/DCA1. Paper mode also skips the released 3x3 opening: that opening erases
one-pixel vessels, so an empty prediction has exactly zero L_C on
CrackTree/CREMI/CORN. Without the opening, empty predictions are penalized
(DRIVE 0.97, CREMI 3.88, CORN 1.84, CrackTree 3.10), but unbounded paper L_C
still explodes on sparse domains (random predictions 11-524). Therefore
paper L_C is an ablation that must be recalibrated with
`--maskvsc-connect-cap` (plus an A/B) before it can be a main-table default.
Release L_C is numerically tiny because sigmoid sharpening makes every pixel
nonzero in `count_nonzero`.

### Stage-1 A/B evidence (local seed 0, 100 epochs, 18/6 split)

All four runs use the same schedule, batch 4 and release ViG graph; only
stage-1 L_C and rotation differ. Best validation Dice:

| L_C mode | rotation 0 | rotation ±30° |
|---|---:|---:|
| release | **0.4612** | **0.4726** |
| paper (unbounded, with released opening) | 0.1133 | 0.1636 |

The paper-mode runs collapse on sparse domains (e.g. CrackTree 0.008,
DCA1 0.000 at the paper-rot0 best), while release L_C reaches CORN 0.405,
CrackTree 0.252 and DCA1 0.430. Rotation is essentially tied under release
L_C, so the main baseline keeps the official flip-only path. These are
100-epoch screening numbers, not final 800-epoch results.

The paper only names **ViG-Ti** and gives no block/width/K table. The release
uses 16 blocks, K=9 and feature width 256; this port keeps that architecture
because the paper does not provide enough detail to replace it.

## ViG decode: official release semantics

The released stage-2 runs as shipped with `patch_size=16`. Its decode is:

```python
x = features.unsqueeze(2).view(B, N, patch, patch)
x = self.predictor(x)          # width N = (img_size / patch)^2
```

With `num_channels=1`, every token has `patch*patch` features, so this folds
each token vector into a `patch x patch` map and uses the N token maps as
channels. The three released variants line up with this:

| Variant | input | patch | N | predictor width |
|---|---|---|---|---|
| Fundus | 512 | 16 | 1024 | 1024 |
| OCTA500 | 384 | 16 | 576 | 576 |
| 2PFM | 1024 | 16 | 4096 | 4096 |

`decode='release'` (default) reproduces this exactly. The official Fundus
ViG has **23,841,764** parameters; with the 7,764,065-parameter UNet the full
model is **31,605,829** parameters and the checkpoint is about 127 MB. The
paper-Euclidean KNN changes graph construction only, not parameter counts.

Release-parity check against the pristine `3rd/MaskVSC/model/gcn.py`: with
`decode='release'` and `graph_metric='release'`, the state-dict keys and
shapes are identical, and with the same weights the forward output is
bit-identical (max abs difference 0.0) for Fundus 512/16 and OCTA 384/16.
The default `graph_metric='paper'` intentionally differs from release.

An earlier port misread this decode as a channel error, transposed the token
tensor into a real `[B, C, side, side]` grid and used patch 32. That variant
(371.6M ViG / 379.36M full) remains selectable as
`--maskvsc-vig-decode coherent` for an explicit ablation, but it is **not**
the released architecture and must not be reported as the official model.

The unused `ViGSeg_2PFM` release head remains internally inconsistent (its
second convolution expects `patch^2` while the first emits 1024 channels);
it is not exercised by the adapter. Fundus is the only stage-2 variant used.

## Two-stage runtime

- Stage 1: official UNet + masked-vessel corruption + BCE + released
  Connect_Loss; release horizontal-flip augmentation; 600 epochs by default.
  Paper L_C/rotation are explicit ablations.
- Stage 2: stage 1 is frozen (`requires_grad=False`) and forced to `eval()`;
  ViG is trained with BCE for 200 epochs on clean training images and the
  frozen stage-1 probability maps. No stage-2 augmentation (release loader;
  the paper does not specify it).
- Stage-2 labels use the official `ImageDataset_GCN_Fundus` soft,
  PIL-resized masks (no binarization). Stage-1 labels keep the official
  `ImageDataset_Fundus` thresholding.
- Inference: `ViG(Seg(image))`, then threshold and resize to GT resolution.
- Selection uses the 6-image validation Dice split. Upstream evaluates the
  test split every epoch and selects by test Dice; that is not used. The
  stage-1 best is checkpointed at the end of stage 1 (also preserved as
  `stage1_best.pth`), and the final model is the better of the stage-1 and
  stage-2 best validation scores. If stage 1 wins, the checkpoint records
  `use_vig: false` and inference uses the UNet alone. This avoids a stage-2
  regression replacing a better stage-1 model.
- The custom training loop has no prediction implementation of its own;
  `--stage predict` dispatches to the shared prediction/evaluation path after
  restoring the checkpoint's `use_vig`, ViG, loss, and augmentation config.
- A single pooled 512 ViG is used across domains. Upstream trains separate
  Fundus/OCTA/2PFM completion models per dataset; unseen/novel protocol
  domains have no per-domain training data, so a pooled model is the only
  protocol-consistent adaptation.
- The 800-epoch shared budget is split 600 stage-1 + 200 stage-2 by default.
  The paper's schedule is 200+200; `--maskvsc-stage1-epochs 200` gives the
  paper schedule (400 total).
- The custom two-stage loop does not implement `--resume`, gradient
  accumulation, or gradient clipping.
- Older checkpoints carry no new selector keys: a missing `vig_decode` is
  inferred as `coherent`; missing `vig_graph`/`connect_loss`/
  `rotation_degrees` are inferred as release graph, release L_C and
  flip-only augmentation. Stage-1-only checkpoints contain no ViG weights
  and must be retrained.
- No official ViG weights are released; the ViG is trained from scratch.
