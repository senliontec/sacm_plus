# Deep Closing upstream record

## Source

- Repository: `5k5000/DeepClosing`
  (https://github.com/5k5000/DeepClosing)
- Local reference: `3rd/DeepClosing`
- HEAD: `dbeb0eaab0687539fd6ad7d2dee46896fdb2a4e5` (2024-05-21)
- License: MIT (copyright Qian Wu, 2024); copied as `LICENSE`.
- Paper: Qian Wu, Yufei Chen, Wei Liu, Xiaodong Yue, Xiahai Zhuang,
  "Deep Closing: Enhancing Topological Connectivity in Medical Tubular
  Segmentation", IEEE Transactions on Medical Imaging, 2024.

The local and upstream HEAD are identical.  The upstream repository is a
partial research release; its README explicitly says more information will
be provided after acceptance.  It contains only DRIVE training code and a
core prototype, not a complete inference/evaluation pipeline.

## What was preserved

The paper's core algorithm is preserved:

1. **Masked Image Modeling (MIM) shape prior**
   - random block masking with patch sizes in `[2, 8]` and mask ratio in
     `[0.1, 0.2]`;
   - Recurrent TubeScaler using recurrent simple-point erosion;
   - reconstruction MSE plus topology-aware MSE at non-simple points
     within the masked region (paper Eq. 4, alpha = 0.1).
2. **Deep Dilation**: apply the shape prior to a pre-segmentation mask,
   sigmoid + threshold (default 0.2).
3. **Simple Component Erosion**: label the positive residue
   `T_dd - T`, erode simple points recurrently, and retain only components
   that survive; the original prediction is never removed.
4. **Simple Points**: the 512-entry 3x3 lookup table, 8-connected
   foreground / 4-connected background criterion, four checkerboard order
   masks, and recurrent parallel erosion.

## Why this is a clean reimplementation

The upstream `DeepClosing.py` inference method cannot run as released:

- it references an undefined `transform` for tensor inputs;
- it asserts `T.unique() == torch.tensor([0, 1])`, which raises for valid
  binary tensors with both values;
- it calls `transform.inverse(...)`, but the local MONAI `Compose` object
  has no such inverse API;
- it assumes batch size 1;
- it references undefined `self.roi_size` in fallback branches;
- it hardcodes `/root/DeepClosing-private/Data/Drive`;
- it only provides DRIVE configs and no test/evaluation entry point.

We therefore implemented the core operations in pure PyTorch/scipy and
documented every deviation.  No file under `3rd/` is imported or modified.

## Deviations

1. **No MONAI or PyTorch Lightning dependency.**  The shape prior is a
   compact 2D U-Net; the default channels are
   `(16, 32, 64, 128, 256)` rather than MONAI's
   `(32, 64, 128, 256, 512)` to fit 512x512 training on the 18-mask
   protocol.  Channels are stored in the checkpoint and configurable.
2. **Resolution.**  The official implementation trains/evaluates with
   224x224 crops and sliding-window inference.  This port uses a
   configurable square working resolution (default 512) and nearest
   resize back to the original prediction size.  This is a task
   adaptation for thin structures; it avoids a lossy 224 crop and is
   documented rather than hidden.
3. **Mask generator.**  The official block mask applies a random affine
   transform.  We retain block masking and use flips/90-degree rotations
   in the training script; random affine masking is omitted.
4. **Recurrent TubeScaler region.**  The paper writes `RSPE(L, 0, k)`,
   while the released code calls `RSPE(T=x, M_T=x, max_K=k_r)`.  We follow
   the executable code: the whole foreground is the erosion region.
5. **Topology-aware loss.**  We implement paper Eq. 3 directly:
   non-simple points remaining in the masked region after RSPE receive
   MSE supervision.  The released code's `critical_points` computation is
   approximate and differs in detail.
6. **Simple-point backend.**  scipy.ndimage.label with 8-connected
   foreground / 4-connected background structures replaces
   scikit-image label; the component-count criterion is identical.
7. **Inference safety.**  Empty masks are returned unchanged; the refiner
   only adds non-simple proposed components and never removes original
   foreground.  A batched call loops over samples because connected
   component labelling is per-image.
8. **Training data.**  The shape prior is trained only on
   `<train_root>/train/masks`; `<val_root>/val/masks` is used for
   checkpoint selection.  Test masks are never read by the training
   script.  The official four-dataset protocol is not copied.
9. **No official weights.**  The upstream repository releases no shape
   prior checkpoint; training starts from random initialization.

## Intended use

Deep Closing is a **post-processing refiner**, not a segmentation
backbone.  Train the shape prior once on the 18 train masks, then apply it
to the predictions of any model (or to SACM-X via
`evaluate.py --deep_closing_checkpoint`).  Report it as
`<method> + Deep Closing`, not as a standalone baseline.
