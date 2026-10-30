"""Skeleton Recall loss for thin curvilinear structures (ECCV 2024).

This is a project adaptation of the loss released with:

    Kirchhoff et al., "Skeleton Recall Loss for Connectivity Conserving and
    Resource Efficient Segmentation of Thin Tubular Structures", ECCV 2024.
    https://github.com/MIC-DKFZ/Skeleton-Recall (Apache-2.0)

The upstream loss is computed on a *tubed* ground-truth skeleton that the
official data pipeline appends as a second segmentation channel.  We do not
modify the dataset (the benchmark's transforms are shared with every other
model); instead the skeleton is computed online from the same augmented GT
used by the mask losses:

    1. ``skimage.morphology.skeletonize`` on the binary GT,
    2. an optional 2-pixel cross dilation (the official "tube"),
    3. clip the tube to the GT foreground, exactly as the upstream
       ``SkeletonTransform`` does,
    4. soft recall of the prediction over that skeleton.

Deviation from upstream: it returns ``1 - recall`` (a positive loss to be
minimized, like the project's clDice loss) rather than the negated recall.
The smoothing and the soft-recall formula are otherwise unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage as ndi
from skimage.morphology import dilation, skeletonize

# Same footprint as upstream SkeletonTransform.get_tube_footprint(2).
_TUBE_FOOTPRINT = ndi.iterate_structure(
    ndi.generate_binary_structure(2, 1), 2)


def tubed_skeletonize(mask, do_tube=True):
    """Return the (optionally tubed) skeleton of a binary 2-D mask.

    Args:
        mask: 2-D array-like, foreground = non-zero.
        do_tube: dilate the skeleton by the official 2-px cross footprint.
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)

    skel = skeletonize(mask)
    if do_tube:
        skel = dilation(skel.astype(np.uint8),
                        footprint=_TUBE_FOOTPRINT).astype(bool)
        # Upstream multiplies by the segmentation, i.e. the tube never
        # extends outside the ground-truth foreground.
        skel = skel & mask
    return skel


class SoftSkeletonRecallLoss(nn.Module):
    """Soft recall of the prediction on the ground-truth skeleton.

    Args:
        do_tube: use the official 2-px tubed skeleton (default True).
        smooth: smoothing term added to numerator and denominator, matching
            the upstream soft-recall formula (upstream default: 1e-5; the
            project uses 1.0 so tiny structures still give a stable signal).
        from_logits: apply sigmoid to the prediction internally (the project
            passes raw mask logits).

    The target skeleton is computed without gradients.  Empty GT masks yield
    a recall of 1 and therefore zero loss, matching the smoothed upstream
    computation.
    """

    def __init__(self, do_tube=True, smooth=1.0, from_logits=True):
        super().__init__()
        self.do_tube = bool(do_tube)
        self.smooth = float(smooth)
        self.from_logits = bool(from_logits)

    def forward(self, pred, target):
        if pred.shape[1] != 1:
            raise ValueError(
                f"SoftSkeletonRecallLoss expects one foreground channel, "
                f"got {pred.shape[1]}")
        if pred.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                "prediction and target must have the same spatial size, "
                f"got {tuple(pred.shape[-2:])} vs {tuple(target.shape[-2:])}")
        if target.shape[1] != 1:
            raise ValueError(
                f"target must have one channel, got {target.shape[1]}")

        probs = torch.sigmoid(pred) if self.from_logits else pred

        with torch.no_grad():
            skeletons = []
            target_np = target.detach().float().cpu().numpy()
            for i in range(target_np.shape[0]):
                skel = tubed_skeletonize(target_np[i, 0] > 0.5,
                                         do_tube=self.do_tube)
                skeletons.append(
                    torch.from_numpy(skel.astype(np.float32)))
            skel = torch.stack(skeletons, dim=0).unsqueeze(1)
            skel = skel.to(device=probs.device, dtype=probs.dtype)

        axes = tuple(range(1, probs.dim()))
        inter = (probs * skel).sum(dim=axes)
        denom = skel.sum(dim=axes)
        recall = (inter + self.smooth) / torch.clamp(
            denom + self.smooth, min=1e-8)
        return 1.0 - recall.mean()
