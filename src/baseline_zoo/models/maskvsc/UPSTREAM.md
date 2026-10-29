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

The paper and the released code disagree in several places. Defaults follow
the paper; the released behavior is retained behind explicit switches for
ablation/parity.

| Topic | Paper | Release code | Default here |
|---|---|---|---|
| Connectivity loss | Eq.1 divides by foreground pixels; Eq.2 S is a mean over N | `count_nonzero(lab_after_sigmoid)` counts every pixel | Paper-intended scale `|S(Y)-S(Y_G)|*N/#foreground`; `--maskvsc-connect-loss release` for the released arithmetic |
| Stage-1 augmentation | random rotation ±30° + horizontal flip | horizontal flip only; no rotation anywhere | Paper rotation + flip; `--maskvsc-rotation-degrees 0` for release |
| Stage-2 augmentation | not separately specified | none in `ImageDataset_GCN_*` | none (release) |
| ViG KNN | Euclidean distance | dot-product top-k | Paper Euclidean; `--maskvsc-vig-graph release` for the release graph |
| Stage-1 freeze in stage 2 | explicitly frozen | not `eval()`; only absent from the optimizer, so BatchNorm buffers drift | Paper frozen + `eval()` |
| Input size | per dataset: 512 DRIVE/RITE, 576 STARE, 1024 LES-AV/2PFM, 384/304 OCTA-500, 304 ROSE | Fundus 512, OCTA 384, 2PFM native | unified 512 benchmark input |
| Corruption | Fundus SD20/7×7, OCTA SD100/3×3, 2PFM SD20/11×11 | matches per-modality classes | unified SD20/7×7 pooled protocol |
| Epochs | 200 per training step (200+200) | 200+200 | 600+200 under the shared 800-epoch budget |

The printed L_C equations are themselves scale-degenerate: Eq.2 normalizes S
by N while Eq.1 divides by foreground pixels, so the literal combination is
also ~1e-6 and numerically inert. Paper mode restores the compensating N
factor, matching the magnitude implied by the paper's L_C ablation; this is
a documented reconstruction, not a literal transcription. On a real DRIVE
512 mask with an imperfect prediction: BCE 0.307, release L_C 1.5e-7
(gradient 1.9e-8), paper L_C 0.665 (gradient 8.0e-2). The release value is
tiny because sigmoid sharpening makes every pixel nonzero in
`count_nonzero`.

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

- Stage 1: official UNet + masked-vessel corruption + BCE + paper-scaled
  Connect_Loss; paper ±30° rotation + horizontal flip; 600 epochs by default.
- Stage 2: stage 1 is frozen (`requires_grad=False`) and forced to `eval()`;
  ViG is trained with BCE for 200 epochs on clean training images and the
  frozen stage-1 probability maps. No stage-2 augmentation (release loader;
  the paper does not specify it).
- Stage-2 labels use the official `ImageDataset_GCN_Fundus` soft,
  PIL-resized masks (no binarization). Stage-1 labels keep the official
  `ImageDataset_Fundus` thresholding.
- Inference: `ViG(Seg(image))`, then threshold and resize to GT resolution.
- Stage-2 selection uses the 6-image validation Dice split. Upstream evaluates
  the test split every epoch and selects by test Dice; that is not used.
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
