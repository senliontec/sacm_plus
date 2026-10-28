"""A2SAM structural prompt generator (SPG).

The paper's SPG turns a probabilistic U-Net likelihood map into a Morse
complex, prunes low-persistence branches, samples the retained branches and
fuses the resulting structural map with the coarse likelihood and the
structural uncertainty into a dense SAM prompt.  This module implements that
pipeline: ``topology.extract_structural_points`` supplies the pruned branch
samples, a small learnable branch gate scores each sample, the samples are
rasterized into a Gaussian structural map, and a Conv-BN-LeakyReLU fusion
produces the dense prompt.

The released repository's ``prompt_discriminator.PromptGenerator`` only
concatenates precomputed uncertainty/logit maps and its output is detached
from the loss; this port replaces it with the trainable structural pipeline
described in the paper and keeps the released Conv-BN-LeakyReLU fusion
pattern.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .topology import StructuralPoints, extract_structural_points


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    grid = coords[:, None] ** 2 + coords[None, :] ** 2
    kernel = torch.exp(-grid / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size)


class StructuralPromptGenerator(nn.Module):
    """Topology-aware dense prompt from likelihood and uncertainty maps."""

    def __init__(
        self,
        point_features: int = 7,
        hidden: int = 32,
        prompt_size: int = 256,
        sigma: float = 2.0,
        max_points: int = 512,
        persistence: float = 5.0,
        relative_persistence: float = 0.1,
        stride: int = 3,
    ):
        super().__init__()
        if prompt_size < 8:
            raise ValueError('prompt_size must be at least 8')
        self.prompt_size = int(prompt_size)
        self.sigma = float(sigma)
        self.max_points = int(max_points)
        self.persistence = float(persistence)
        self.relative_persistence = float(relative_persistence)
        self.stride = int(stride)
        self.branch_gate = nn.Sequential(
            nn.Linear(point_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.register_buffer(
            'gaussian', _gaussian_kernel(7, max(self.sigma, 0.5)))

    def prepare(self, prob: torch.Tensor,
                uncertainty: torch.Tensor) -> list[StructuralPoints]:
        """CPU topology extraction for every image in the batch."""
        prob_np = prob.detach().float().cpu().numpy()
        unc_np = uncertainty.detach().float().cpu().numpy()
        out = []
        for index in range(prob_np.shape[0]):
            out.append(extract_structural_points(
                prob_np[index, 0], unc_np[index, 0],
                persistence=self.persistence,
                relative_persistence=self.relative_persistence,
                stride=self.stride,
                max_points=self.max_points,
            ))
        return out

    def _rasterize(self, structural: list[StructuralPoints],
                   height: int, width: int) -> torch.Tensor:
        size = self.prompt_size
        device = self.branch_gate[0].weight.device
        maps = []
        for points in structural:
            if points.empty:
                maps.append(torch.zeros(
                    (1, 1, size, size), device=device))
                continue
            features = torch.from_numpy(points.features).to(device)
            saliency = torch.from_numpy(points.saliency).to(device)
            gate = torch.sigmoid(
                self.branch_gate(features)).squeeze(1)
            weights = gate * saliency
            xs = torch.from_numpy(points.points[:, 0]).to(device)
            ys = torch.from_numpy(points.points[:, 1]).to(device)
            xs = (xs / max(width - 1, 1) * (size - 1)).round().long()
            ys = (ys / max(height - 1, 1) * (size - 1)).round().long()
            xs = xs.clamp(0, size - 1)
            ys = ys.clamp(0, size - 1)
            flat = torch.zeros((size * size,), device=device)
            flat.index_add_(0, ys * size + xs, weights)
            canvas = flat.view(1, 1, size, size)
            canvas = F.conv2d(
                canvas, self.gaussian,
                padding=self.gaussian.shape[-1] // 2)
            normalizer = canvas.amax(dim=(2, 3), keepdim=True).clamp_min(
                1e-6).detach()
            maps.append(canvas / normalizer)
        return torch.cat(maps, dim=0)

    def render(self, structural: list[StructuralPoints],
               prob: torch.Tensor,
               uncertainty: torch.Tensor) -> torch.Tensor:
        """Render a dense prompt from cached topology samples."""
        if len(structural) != prob.shape[0]:
            raise ValueError('structural list and batch size differ')
        height, width = prob.shape[-2:]
        structural_map = self._rasterize(structural, height, width)
        size = (self.prompt_size, self.prompt_size)
        prob_small = F.interpolate(
            prob.float(), size=size, mode='bilinear', align_corners=False)
        unc_small = F.interpolate(
            uncertainty.float(), size=size, mode='bilinear',
            align_corners=False)
        return self.fuse(torch.cat(
            [structural_map, unc_small, prob_small], dim=1))

    def forward(self, prob: torch.Tensor,
                uncertainty: torch.Tensor) -> torch.Tensor:
        structural = self.prepare(prob, uncertainty)
        return self.render(structural, prob, uncertainty)
