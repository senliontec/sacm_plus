"""Shared helpers for the metrics package."""

import numpy as np


def to_numpy(x):
    """Convert tensor/array to numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def binarize(x, threshold=0.0):
    """Binarize logits/probabilities to {0, 1}."""
    return (x > threshold).astype(np.uint8)


def _bool(m):
    # 与 medpy 逐字等价:atleast_1d(astype(bool))——非零即真
    # (含负值/NaN 的极端输入也与官方行为一致)
    return np.atleast_1d(np.asarray(m).astype(bool))


def zhang_suen_thinning(mask):
    """Zhang-Suen morphological thinning (bundled fallback for skimage)."""
    img = (_bool(mask)).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            pad = np.pad(img, 1, mode="constant")
            p2 = pad[:-2, 1:-1]
            p3 = pad[:-2, 2:]
            p4 = pad[1:-1, 2:]
            p5 = pad[2:, 2:]
            p6 = pad[2:, 1:-1]
            p7 = pad[2:, :-2]
            p8 = pad[1:-1, :-2]
            p9 = pad[:-2, :-2]
            seq = [p2, p3, p4, p5, p6, p7, p8, p9]
            a = sum(((p == 0) & (q == 1)) for p, q in zip(seq, seq[1:] + seq[:1]))
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            c1 = (b >= 2) & (b <= 6)
            c2 = a == 1
            c3 = ((p2 * p4 * p6) == 0) if step == 0 else ((p2 * p4 * p8) == 0)
            c4 = ((p4 * p6 * p8) == 0) if step == 0 else ((p2 * p6 * p8) == 0)
            m = (img == 1) & c1 & c2 & c3 & c4
            if m.any():
                img[m] = 0
                changed = True
    return img.astype(bool)


try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def skeletonize(mask):
    """skimage.morphology.skeletonize (the official clDice metric's
    function) when available; bundled Zhang-Suen fallback otherwise."""
    if _HAS_SKIMAGE:
        return _skimage_skeletonize(_bool(mask))
    return zhang_suen_thinning(mask)
