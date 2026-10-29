# BCUNet

- Upstream: https://github.com/ZX-ECJTU/BCU-Net
- Local snapshot: `3rd/BCU-Net`, HEAD
  `008ff9292097791a3f12c2776aea398f71b7c996`.
- Paper: Zhang et al., "BCU-Net: Bridging ConvNeXt and U-Net for medical
  image segmentation", Computers in Biology and Medicine 159 (2023) 106960.
- License: no LICENSE file in the vendored snapshot; add the upstream
  license before redistributing this copy.
- Files copied:
  - `code/core/models.py` -> `model.py` (relative imports, configurable
    branch-fusion weights)
  - `code/core/convnext.py` -> `convnext.py` (unchanged)
  - `code/core/unet_parts.py` -> `unet_parts.py`
- Added files:
  - `augmentation.py`: deterministic offline 18-variant training dataset.
- Runtime consumer: `baseline_zoo.adapters.bcunet.BCUNetAdapter`.

## Paper profile implemented by default

- Paper method: MRL/RecallCE. L1 supervises ConvNeXt, L3 supervises U-Net,
  L2 supervises the average-fused output; total loss is L1+L2+L3
  (paper Eq. 1/3 and Fig. 2).
- Fusion: `0.5 * U-Net + 0.5 * ConvNeXt`; paper Fig. 8/§4.1 states
  alpha=beta=0.5 and defines them as branch weights. The paper's Eq. 2
  (`L2 = alpha L1 + beta L3`) conflicts with Fig. 2; this port follows the
  figure and Fig. 8.
- U-Net upsampling: bilinear, as described in §3/Fig. 2.
- Both branches return raw 2-class logits; the paper's MRL uses
  softmax/RecallCE.
- Training: Adam 1.5e-4, batch 2, 100 epochs (paper §4.1/§4.5.3).
- Augmentation: 18 deterministic offline variants per training image,
  including the identity variant (rotation, flips, brightness, Gaussian
  noise, Gaussian blur, and combinations). The paper does not say whether
  the original counts inside the 18; this port uses 18 total. The paper
  gives no amplitudes/probabilities; this is a documented reconstruction,
  not an official parameter set.
- Inputs: BGR [0,255], INTER_AREA resize, no normalization and no other
  preprocessing, matching the release data loader and paper §4.1.

## Paper text vs release architecture

- The paper text says the ConvNeXt branch has three continuous upsampling
  blocks, while `models.py` has four (`up4`..`up1`); the fourth `up1` has no
  skip and is required to recover 32x32 to 512x512. The port follows the
  release.
- The paper text says a U-Net block contains one BN/ReLU, while Fig. 2 and
  the released `DoubleConv` contain two. The port follows Fig. 2/code.
- Algorithm 1 lists training ConvNeXt (L1) and U-Net (L3) before bridging
  (L2), then joint training. The release `main.py` jointly optimizes
  L1/L2/L3 from the first batch and provides no stage epochs. The port
  follows the joint code path.
- The paper scopes the 18x augmentation to retinal vessel datasets. The
  port's `retinal` scope applies the transforms only to
  CHASEDB1/DRIVE/STARE/DRHAGIS/HRF/LES-AV and repeats other domains with
  the identity variant to preserve the pooled 18/6 balance;
  `--bcunet-augment-scope all` is a task adaptation.

## Necessary repairs and documented deviations

1. `OutConv` calls an undefined `upconv2x2` and depends on missing
   `warmlearnrate.py`; it also applies Sigmoid. The paper MRL uses
   softmax/RecallCE, so the port uses a full-resolution 1x1 logits
   projection (no Sigmoid). This is a reconstruction, not official code.
2. The released bilinear `Up` branch is broken: `Upsample` keeps
   `in_channels` but concatenation with a skip of `in_channels // 2`
   produces `1.5 * in_channels`, while `DoubleConv` expects `in_channels`.
   The port constructs the bilinear conv with the actual concatenated
   width. The release ConvTranspose path is unchanged and selectable with
   `--bcunet-convtranspose`.
3. `RecallCrossEntropy` in the release hardcodes `.cuda()` and has
   fragile ignore-index indexing. The port implements paper Eq. 1
   device-safely on raw logits; do not feed it log-softmax probabilities.
4. The release `main.py` uses CE and comments out RecallCE; `ioutrain.py`
   uses RecallCE but feeds log-softmax into `F.cross_entropy`, a
   double-log-softmax bug. The port uses the paper MRL by default and
   keeps `--bcunet-loss ce` for the CE ablation.
5. The release fuses outputs as `0.4*U-Net + 0.6*ConvNeXt`; this conflicts
   with the paper's alpha=beta=0.5. Use
   `--bcunet-fusion-unet 0.4 --bcunet-fusion-convnext 0.6` for release
   parity.
6. The release has no augmentation code; the paper describes 18x offline
   augmentation. Use `--bcunet-augment-variants 1` for the release/no-aug
   ablation. The exact paper recipe is unavailable.
7. No official BCU-Net trained weights are released; only the ConvNeXt
   ImageNet-22k backbone is initialized. The port trains from scratch on
   the shared 3-shot split and selects by validation Dice instead of the
   paper's five-fold/test protocol.
