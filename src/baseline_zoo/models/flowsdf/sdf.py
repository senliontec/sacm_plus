"""Signed-distance targets, Euler ODE sampling and tiled inference for FlowSDF.

The official repository precomputes SDF masks with the companion
``generative-segmentation-sdf`` notebook and samples with ``torchdiffeq``.
This port computes the truncated signed distance on the fly with SciPy's
exact Euclidean distance transform and integrates the probability-flow ODE
with a local Euler loop, so no optional dependency is required.

Convention (matching FlowSDF): negative inside the object, positive outside,
truncated at ``delta`` pixels and divided by ``delta``.  A binary mask is
recovered with ``sdf <= threshold``; the official sampler uses
``threshold = 3 * 1e-2``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def signed_distance_map(mask: np.ndarray, delta: float = 15.0) -> np.ndarray:
    """Return a truncated signed distance map in ``[-1, 1]``.

    ``mask`` is a boolean array at GT resolution.  The mask is padded before
    the distance transform so objects touching the image border still have a
    well-defined outside distance.  Empty masks map to all +1 (background)
    and full masks to all -1 (foreground).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f'expected a 2-D mask, got {mask.shape}')
    if delta <= 0:
        raise ValueError(f'delta must be positive, got {delta}')
    if not mask.any():
        return np.ones(mask.shape, dtype=np.float32)
    if mask.all():
        return -np.ones(mask.shape, dtype=np.float32)

    pad = int(np.ceil(delta)) + 1
    padded = np.pad(mask, pad, mode='constant', constant_values=False)
    inside = distance_transform_edt(padded)[pad:-pad, pad:-pad]
    outside = distance_transform_edt(~padded)[pad:-pad, pad:-pad]
    sdf = np.where(mask, -inside, outside)
    sdf = np.clip(sdf, -delta, delta) / delta
    return sdf.astype(np.float32)


def tile_starts(length: int, patch: int, stride: int) -> list[int]:
    """Patch origins covering ``length`` with the last patch flush to the edge."""
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def euler_sample(
    model: torch.nn.Module,
    image: torch.Tensor,
    noise: torch.Tensor,
    *,
    sigma_min: float = 1e-5,
    nfe: int = 4,
) -> torch.Tensor:
    """Integrate the FlowSDF probability-flow ODE with Euler steps.

    This reproduces the official sampler structure: the integration variable
    ``m`` is blended with the fixed initial noise at each time,

        m_input = (1 - (1 - sigma_min) * t) * noise + t * m,

    and the network velocity is used as ``dm/dt``.  The official code passes
    ``torch.linspace(0, 1, 4)`` to ``odeint`` (three Euler steps) while its
    config labels the setting "4 NFE"; this port implements ``nfe`` equal
    steps, matching Algorithm 2 of the paper.

    ``image`` is ``[K,3,H,W]`` and ``noise`` is ``[K,1,H,W]``.  Returns the
    final SDF sample(s) with the same shape as ``noise``.
    """
    if nfe < 1:
        raise ValueError(f'nfe must be >= 1, got {nfe}')
    if image.shape[0] != noise.shape[0]:
        raise ValueError('image and noise batch sizes differ')
    m = noise
    batch = noise.shape[0]
    step = 1.0 / nfe
    for index in range(nfe):
        t_value = index * step
        t = torch.full(
            (batch, 1), t_value, device=noise.device, dtype=noise.dtype
        )
        alpha = 1 - (1 - sigma_min) * t_value
        m_input = alpha * noise + t_value * m
        velocity = model(m_input, t, img_cond=image)
        m = m + step * velocity
    return m


def _prepare_image(
    image: torch.Tensor, patch: int
) -> tuple[torch.Tensor, tuple[int, int], bool]:
    """Resize an image so both sides are at least ``patch``; return resized flag."""
    height, width = image.shape[-2:]
    if height >= patch and width >= patch:
        return image, (height, width), False
    scale = max(patch / height, patch / width)
    new_h = max(patch, int(round(height * scale)))
    new_w = max(patch, int(round(width * scale)))
    resized = F.interpolate(
        image.unsqueeze(0), size=(new_h, new_w), mode='bilinear',
        align_corners=False
    ).squeeze(0)
    return resized, (height, width), True


@torch.no_grad()
def predict_whole(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    size: int = 128,
    k_samples: int = 1,
    nfe: int = 4,
    sigma_min: float = 1e-5,
    seed: int = 0,
) -> torch.Tensor:
    """Predict one dense SDF map from the whole image resized to ``size``.

    This is the official GlaS inference path (entire 128x128 image), used for
    the per-epoch validation pass because it needs one ODE solve instead of a
    tile grid.  Final test prediction uses :func:`predict_tiled`.
    """
    if k_samples < 1:
        raise ValueError(f'k_samples must be >= 1, got {k_samples}')
    height, width = image.shape[-2:]
    work = F.interpolate(
        image.unsqueeze(0), size=(size, size), mode='bilinear',
        align_corners=False
    )[0]
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed) & 0xFFFFFFFF)
    noise = torch.randn(
        (k_samples, 1, size, size), generator=generator, dtype=torch.float32
    ).to(image.device)
    samples = euler_sample(
        model,
        work.unsqueeze(0).repeat(k_samples, 1, 1, 1),
        noise,
        sigma_min=sigma_min,
        nfe=nfe,
    )
    sdf = samples.mean(dim=0)
    return F.interpolate(
        sdf[None], size=(height, width), mode='bilinear', align_corners=False
    )[0, 0]


@torch.no_grad()
def predict_tiled(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    patch: int = 128,
    stride: int = 64,
    k_samples: int = 1,
    nfe: int = 4,
    sigma_min: float = 1e-5,
    seed: int = 0,
    tile_chunk: int = 4,
) -> torch.Tensor:
    """Predict a dense SDF map by averaging overlapping 128x128 patches.

    ``image`` is a normalized ``[3,H,W]`` tensor.  The returned map is at the
    original resolution.  Each tile is sampled ``k_samples`` times with
    independent noise and averaged (MMSE, as in the paper); overlapping tile
    predictions are averaged in SDF space before thresholding.
    """
    if patch <= 0 or patch % 32 != 0:
        raise ValueError(f'patch must be a positive multiple of 32, got {patch}')
    if stride <= 0:
        raise ValueError(f'stride must be positive, got {stride}')
    if k_samples < 1:
        raise ValueError(f'k_samples must be >= 1, got {k_samples}')
    if tile_chunk < 1:
        raise ValueError(f'tile_chunk must be >= 1, got {tile_chunk}')

    work, original, resized = _prepare_image(image, patch)
    height, width = work.shape[-2:]
    tops = tile_starts(height, patch, stride)
    lefts = tile_starts(width, patch, stride)
    accumulator = torch.zeros((height, width), device=image.device,
                              dtype=torch.float32)
    counts = torch.zeros((height, width), device=image.device,
                         dtype=torch.float32)

    positions = [(top, left) for top in tops for left in lefts]
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed) & 0xFFFFFFFF)

    for start in range(0, len(positions), tile_chunk):
        chunk = positions[start:start + tile_chunk]
        images = torch.stack(
            [work[:, top:top + patch, left:left + patch] for top, left in chunk]
        )
        images = images.repeat_interleave(k_samples, dim=0)
        noise = torch.randn(
            (images.shape[0], 1, patch, patch),
            generator=generator, dtype=torch.float32,
        ).to(image.device)
        samples = euler_sample(
            model, images, noise, sigma_min=sigma_min, nfe=nfe
        )
        samples = samples.view(len(chunk), k_samples, 1, patch, patch)
        tile_sdf = samples.mean(dim=1)[:, 0]
        for (top, left), value in zip(chunk, tile_sdf):
            accumulator[top:top + patch, left:left + patch] += value
            counts[top:top + patch, left:left + patch] += 1

    sdf = accumulator / counts.clamp_min(1)
    if resized:
        sdf = F.interpolate(
            sdf[None, None], size=original, mode='bilinear',
            align_corners=False
        )[0, 0]
    return sdf
