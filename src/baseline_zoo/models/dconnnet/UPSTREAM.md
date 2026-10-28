# DconnNet upstream record

## Sources

| Component | Upstream | Version |
|---|---|---|
| Paper | "Directional Connectivity-based Segmentation of Medical Images", CVPR 2023 | 2023 |
| Code | `https://github.com/Zyun-Y/DconnNet` (local `3rd/DconnNet`) | default branch |
| Paper parsing | `parsing/[2023 cvpr] Directional connectivity-based segmentation of medical images` | local |
| License | the release has **no LICENSE file**; verify redistribution rights before publishing | upstream |

## Vendored files

| Release | Port |
|---|---|
| `model/DconnNet.py` | `model.py` (imports made package-relative; `torchsummary` and the unrelated torchvision import removed; `resnet34(pretrained=False)`) |
| `model/attention.py` | `attention.py` (verbatim) |
| `model/gap.py` | `gap.py` (verbatim) |
| `model/resnet.py` | `resnet.py` (only the ResNet-34 path used by DconnNet; network URL/torchsummary removed; release's `conv1_1` typo in the unused `forward` fixed) |
| `connect_loss.py` | `connect_loss.py` (device-aware; batch-preserving; empty-edge guard) |

The binary model uses `num_class=1`: the network emits eight directional
connectivity logits plus an auxiliary eight-channel map at the input
resolution.  The CHASE objective is `BCE + connectivity BCE + edge loss +
Dice`; the release also computes the auxiliary loss with weight 0.3 during
training, which the adapter preserves.

## Upstream defects fixed during the port

- `connectivity_matrix` called `squeeze()`, deleting the batch dimension when
  `B=1`; the port keeps `[B,8,H,W]`.
- The loss and voting helpers hard-coded `.cuda()`; the port uses the input
  device, so CPU tests and non-zero GPU ids work.
- `ResNet.forward` referenced undefined `conv1_1`; DconnNet accesses the
  backbone blocks directly, so this path was unused.  The port fixes it.
- The edge-loss denominator had no guard; the port clamps it to 1e-8.
- The release imports `apex`/`torchsummary` only in its trainer; this port
  uses fp32 and does not require either package.

## Protocol adaptations

- Official CHASE script: 960x960, batch 4, Adam lr 3.8e-3, polynomial LR
  power 0.9, 130 epochs.  This port keeps 960, batch 4, lr and polynomial
  shape, but uses the shared 800-epoch budget and validation-Dice selection
  instead of the official test-Dice selection.
- The release reads images with OpenCV (BGR) and normalizes
  `[0,1]*3.2-1.6`; the port converts the protocol RGB loader to BGR and keeps
  that normalization.  Masks are resized and thresholded at 0.5.
- The official HSV shift, perspective shift/scale/rotation, flips and rot90
  are reimplemented on original-resolution protocol pairs; no crop is used.
- Inference follows the release's CHASE path: sigmoid -> 0.5 threshold ->
  reshape to `[B,1,8,H,W]` -> `Bilateral_voting` -> max over directions,
  then resize to the exact GT resolution.
- The ResNet-34 ImageNet checkpoint is optional.  If
  `checkpoints/resnet34-333f7ec4.pth` is present, it is loaded strictly into
  `model.backbone`; otherwise training starts from random initialization and
  the run is marked as a deviation.  No runtime download is attempted.
- SDL is not used: the official CHASE configuration does not enable it.

## Verification

- CPU forward `[1,3,256,256] -> [1,8,256,256]` (+ auxiliary map), 28.4M
  parameters for `num_class=1`.
- GPU one-batch official objective: finite loss and gradients at batch size
  2; batch size 1 is covered by the preserved dimensionality.
- Adapter tests cover connectivity targets/voting, batch-1 loss, BGR data
  preprocessing, polynomial LR, config restoration and GT-size prediction.
- GPU smoke (2 train / 1 val, one epoch at 320): train loss 4.0134, val
  Dice 0.1036, checkpoint saved; 960 batch-4 forward/backward peak is
  6.72 GiB; 20 DRIVE predictions took 1 s and exported 565x584 0/255 masks.
- ResNet-34 state-dict keys/shapes match torchvision's 218-key resnet34,
  and a synthetic checkpoint was loaded strictly by the adapter.
