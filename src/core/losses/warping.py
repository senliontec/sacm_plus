"""Warping loss — strict vendor of the official repository for
"Structure-Aware Image Segmentation with Homotopy Warping" (NeurIPS 2022,
Xiaoling Hu), local mirror 3rd/Warping.

Vendored from:
  warping_loss.py  decide_simple_point_2D (lines 9-42), update_simple_point
                   (lines 69-91), warping_loss 2-D branch (lines 94-143,
                   final CE at line 176)
  utilities.py     softmax_helper (line 9), RobustCrossEntropyLoss
                   (lines 164-172)

How it works (official logic, unchanged): prediction and ground truth are
binarised, false-positive / false-negative pixels are ordered by their
distance-transform value, greedy single-pixel "simple point" flips — flips
that preserve the homotopy of the 3x3 neighbourhood, tested by
4-connectivity of the foreground and 8-connectivity of the background —
warp one mask into the other, and the pixels whose topology still differs
after warping are the CRITICAL POINTS. The loss is the cross-entropy of
the network output masked to those critical points, scaled by their count.

Vendoring method: the 2-D execution path is kept line-by-line; the helpers
below are verbatim except where a deviation is listed.

Deviations (complete list):
1. 2-D ONLY. Everything 3-D needs the cc3d wheel (6/26-connectivity 3-D
   connected components) and is NOT ported: decide_simple_point_3D
   (warping_loss.py:44-67), the ``else`` branch of update_simple_point
   (69-91) and the 5-D branch of warping_loss (144-174). 5-D input raises
   NotImplementedError; the cc3d import is guarded. (The official 5-D
   branch is also internally broken — line 174 writes
   ``critical_points[i,:,:]`` into a ``np.zeros((B,H,W,Z))`` array.)
2. ``import pdb; pdb.set_trace()`` (warping_loss.py:116) dropped — a debug
   leftover that would hang every training run.
3. Device/dtype: ``torch.unsqueeze(torch.from_numpy(critical_points),
   dim=1).cuda()`` (warping_loss.py:176) becomes ``.to(device=y_pred.
   device, dtype=y_pred.dtype)``. The official mask is float64 (``np.zeros``
   default) and silently promotes the whole CE to float64; its values are
   exactly 0.0/1.0, so the cast changes nothing but float rounding, while
   the CPU path and AMP keep working.
4. Empty-patch guard in decide_simple_point_2D: the official slice
   ``gt[x-1:x+2, y-1:y+2]`` is EMPTY when x == 0 or y == 0 (the negative
   start clamps past the stop), and cv2.connectedComponents SEGFAULTS on an
   empty patch (verified with OpenCV 5.0.0 — a hard crash, not an
   exception). We mirror the guard the official 3-D twin already carries
   (``if patch.shape[0] != 0 ...``, warping_loss.py:53): an empty patch is
   skipped, i.e. no flip.
5. utilities.py is vendored only where warping_loss actually calls it:
   softmax_helper (line 9, written as a def instead of a lambda — E731) and
   RobustCrossEntropyLoss (164-172). DC_and_CE_loss and SoftDiceLoss are
   *constructed* at warping_loss.py:107-109 and then never used — dropped
   as dead code.

API bridge: the official function takes [B, C>=2, H, W] softmax logits
(channel 0 = background) plus a [B, 1, H, W] target, and binarises the
prediction with ``argmax(softmax(y_pred), 1)``. The adapter honours the
project contract ([B, 1, H, W] logits + [0, 1] float mask) by building the
official 2-channel form ``cat([zeros, logits])`` — so ``argmax`` is exactly
``sigmoid(logits) > 0.5`` and the final CE sees the same probabilities —
after interpolating to `resolution` (256 by default) when needed.

Notes on official behaviour reproduced as-is (not port artefacts):
  * the target is binarised by uint8 truncation of the numpy copy
    (warping_loss.py:122) and by ``.long()`` for the CE target
    (utilities.py:172), so a soft [0, 1] target is thresholded at 1.0;
  * the CE mask zeroes BOTH ``y_pred`` and the target, so masked-out pixels
    enter the mean CE with uniform (all-zero) logits and contribute ln 2;
  * the loss is scaled by the number of critical points in the batch.

Requires: cv2, scipy — both optional for the package; registration is
skipped when either is missing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register

try:  # official warping_loss.py:1
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

try:  # official warping_loss.py:3 — from scipy import ndimage
    from scipy import ndimage
    _HAS_SCIPY = True
except ImportError:
    ndimage = None
    _HAS_SCIPY = False

try:  # official warping_loss.py:7 — only the (unported) 3-D branch needs it
    import cc3d  # noqa: F401
    _HAS_CC3D = True
except ImportError:
    cc3d = None
    _HAS_CC3D = False


# ---------------------------------------------------------------------------
# utilities.py (loss-relevant parts only)
# ---------------------------------------------------------------------------

def softmax_helper(x):
    """Official utilities.py:9 — ``softmax_helper = lambda x: F.softmax(x, 1)``
    (written as a def so ruff E731 stays clean)."""
    return F.softmax(x, 1)


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """Official utilities.py:164-172, verbatim.

    Compatibility layer: the target is float and carries an extra channel
    dimension, so it is squeezed to the class-index form CrossEntropyLoss
    expects (the ``.long()`` cast truncates soft targets — official).
    """

    def forward(self, input, target):
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


# ---------------------------------------------------------------------------
# warping_loss.py (2-D path)
# ---------------------------------------------------------------------------

def decide_simple_point_2D(gt, x, y):
    """
    decide simple points
    """

    ## extract local patch
    patch = gt[x-1:x+2, y-1:y+2]

    # Deviation 4: the official code has no guard here and cv2 segfaults on
    # the empty patch produced at x == 0 / y == 0. The official 3-D twin
    # (warping_loss.py:53) does guard — same semantics: no local topology
    # to test, so no flip.
    if patch.size == 0:
        return gt

    ## check local topology

    number_fore, _ = cv2.connectedComponents(patch, 4)
    number_back, _ = cv2.connectedComponents(1-patch, 8)

    label = (number_fore-1) * (number_back-1)

    ## flip the simple point
    if (label == 1):
        gt[x, y] = 1 - gt[x, y]

    return gt


def update_simple_point(distance, gt):
    """Official warping_loss.py:69-91. The 3-D ``else`` branch (which calls
    the cc3d-based decide_simple_point_3D) is not ported — see deviation 1."""
    non_zero = np.nonzero(distance)
    # indice = np.argsort(-distance, axis=None)
    indice = np.unravel_index(np.argsort(-distance, axis=None), distance.shape)

    for i in range(len(non_zero[0])):
        x = indice[0][len(non_zero[0]) - i - 1]
        y = indice[1][len(non_zero[0]) - i - 1]

        gt = decide_simple_point_2D(gt, x, y)
    return gt


def warping_loss(y_pred, y_gt):
    """
    Calculate the warping loss of the predicted image and ground truth image
    Args:
        pre:   The likelihood pytorch tensor for neural networks.
        gt:   The groundtruth of pytorch tensor.
    Returns:
        warping_loss:   The warping loss value (tensor)
    """
    ## compute false positive and false negative

    loss = 0
    ce_loss = RobustCrossEntropyLoss()

    if len(y_pred.shape) == 4:
        B, C, H, W = y_pred.shape

        pre = softmax_helper(y_pred)
        pre = torch.argmax(pre, dim=1)
        y_gt = torch.unsqueeze(y_gt[:, 0, :, :], dim=1)
        gt = torch.squeeze(y_gt, dim=1)

        pre = pre.cpu().detach().numpy().astype('uint8')
        gt = gt.cpu().detach().numpy().astype('uint8')

        pre_copy = pre.copy()
        gt_copy = gt.copy()

        critical_points = np.zeros((B, H, W))
        for i in range(B):
            false_positive = ((pre_copy[i, :, :] - gt_copy[i, :, :]) == 1).astype(int)
            false_negative = ((gt_copy[i, :, :] - pre_copy[i, :, :]) == 1).astype(int)

            ## Use distance transform to determine the flipping order
            false_negative_distance_gt = ndimage.distance_transform_edt(gt_copy[i, :, :]) * false_negative  # shrink gt while keep connected
            false_positive_distance_gt = ndimage.distance_transform_edt(1 - gt_copy[i, :, :]) * false_positive  # grow gt while keep unconnected
            gt_warp = update_simple_point(false_negative_distance_gt, gt_copy[i, :, :])
            gt_warp = update_simple_point(false_positive_distance_gt, gt_warp)

            false_positive_distance_pre = ndimage.distance_transform_edt(pre_copy[i, :, :]) * false_positive  # shrink pre while keep connected
            false_negative_distance_pre = ndimage.distance_transform_edt(1-pre_copy[i, :, :]) * false_negative  # grow gt while keep unconnected
            pre_warp = update_simple_point(false_positive_distance_pre, pre_copy[i, :, :])
            pre_warp = update_simple_point(false_negative_distance_pre, pre_warp)

            critical_points[i, :, :] = np.logical_or(np.not_equal(pre[i, :, :], gt_warp), np.not_equal(gt[i, :, :], pre_warp)).astype(int)
    else:
        raise NotImplementedError(
            "warping_loss: only the official 2-D branch (4-D input) is ported; the "
            "5-D branch (warping_loss.py:144-174) needs cc3d "
            f"(importable in this environment: {_HAS_CC3D})."
        )

    # Deviation 3: official is
    #   torch.unsqueeze(torch.from_numpy(critical_points), dim=1).cuda()
    # (float64 + hard-coded CUDA); the mask values are exactly 0.0/1.0.
    critical_mask = torch.unsqueeze(
        torch.from_numpy(critical_points).to(device=y_pred.device, dtype=y_pred.dtype),
        dim=1,
    )

    loss = ce_loss(y_pred * critical_mask, y_gt * critical_mask) * len(np.nonzero(critical_points)[0])
    return loss


class WarpingLossAdapter(nn.Module):
    """Registered adapter ('warping'): [B, 1, H, W] logits + [0, 1] mask in,
    scalar tensor out, `resolution` (256 default) evaluation.

    Bridge: the official warping_loss wants [B, C>=2, H, W] softmax logits
    whose argmax is the binary prediction. ``cat([zeros, logits])`` makes
    that argmax exactly ``sigmoid(logits) > 0.5`` and leaves the final CE
    on the same probabilities as the official 2-channel input.
    A 3-D input ([B, H, W], as core/losses/monitor.py normalises it) is
    accepted and read as a single channel.
    """

    def __init__(self, resolution=256, **kwargs):
        super().__init__()
        self.resolution = resolution

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: [B, 1, H, W] raw logits; target: [B, 1, H, W] in [0, 1]."""
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        assert logits.shape[1] == 1, f"warping expects 1-channel logits, got {tuple(logits.shape)}"

        if self.resolution is not None and logits.shape[-1] != self.resolution:
            logits = F.interpolate(logits, size=(self.resolution, self.resolution),
                                   mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(self.resolution, self.resolution),
                                   mode='bilinear', align_corners=False)

        # Official 2-channel softmax-logit form: softmax(...)[:, 1] == sigmoid(logits)
        y_pred = torch.cat([torch.zeros_like(logits), logits], dim=1)
        return warping_loss(y_pred, target)


# Register only when the hard dependencies (cv2, scipy) are importable —
# mirroring the graceful-degradation pattern of satloss.py / topograph.py.
if _HAS_CV2 and _HAS_SCIPY:
    register('warping')(WarpingLossAdapter)
