"""Widest-Path Reachability Fields (WPRF) for a SAM decoder.

Faithful re-implementation of the WPRF training objective from
"Widest-Path Reachability Fields for Connectivity-Preserving Slender
Structure Segmentation" (2026):

* a lightweight affinity head predicts directed link logits on a coarse
  graph grid;
* a GT-skeleton support graph provides component labels and pair targets;
* a differentiable Max-Min dynamic program turns the affinities into
  k-step widest-path reachability values;
* ``L_edge`` and ``L_reach`` route gradients to connectivity bottlenecks.

Inference uses only the union mask; the head costs nothing at test time.
Deviations from the paper: graph stride 8 at 1024 input (the paper uses 4
at 512), 16 sources per image instead of 32, and the bottleneck-aware
segmentation term is omitted because the SAM scaffold already trains
BCE/Dice plus soft-clDice and skeleton recall.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label
from skimage.morphology import skeletonize

OFFSETS = ((0, 1), (0, -1), (1, 0), (-1, 0),
           (1, 1), (1, -1), (-1, 1), (-1, -1))
REVERSE = (1, 0, 3, 2, 7, 6, 5, 4)


def _shift_to(tensor: torch.Tensor, dy: int, dx: int,
              pad: float = 0.0) -> torch.Tensor:
    """out[v] = tensor[v - (dy, dx)], ``pad`` outside the grid."""
    out = torch.full_like(tensor, float(pad))
    height, width = tensor.shape[-2:]
    src_y = slice(max(0, -dy), height - max(0, dy))
    dst_y = slice(max(0, dy), height - max(0, -dy))
    src_x = slice(max(0, -dx), width - max(0, dx))
    dst_x = slice(max(0, dx), width - max(0, -dx))
    out[..., dst_y, dst_x] = tensor[..., src_y, src_x]
    return out


class AffinityHead(nn.Module):
    """Directed link logits on the graph grid (1/8 of the input)."""

    def __init__(self, in_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1),
            nn.GroupNorm(16, hidden),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.GroupNorm(8, hidden // 2),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden // 2, len(OFFSETS), 1)
        # Start with high link probabilities (A~0.88) so the
        # Max-Min DP has propagating paths at initialization;
        # random logits can underflow sigmoid and explode BCE.
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, 2.0)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        h = self.body(embeddings)
        h = F.interpolate(h, scale_factor=2, mode='bilinear',
                          align_corners=False)
        h = self.up(h)
        return self.out(h)


@torch.no_grad()
def build_support(masks: torch.Tensor, stride: int):
    """Return (support, component ids) on the graph grid.

    ``masks`` is [B,1,H,W] with values in [0,1] at the full input
    resolution.  The support is the max-pooled skeleton; component ids are
    8-connected components of the support grid (0 = background).
    """
    support = []
    comps = []
    for mask in masks.detach().cpu().numpy()[:, 0]:
        skel = skeletonize(mask > 0.5).astype(np.float32)
        t = torch.from_numpy(skel)[None, None]
        pooled = F.max_pool2d(t, stride) > 0
        grid = pooled[0, 0].numpy()
        labels = label(grid, connectivity=2)
        support.append(pooled[0, 0])
        comps.append(torch.from_numpy(labels.astype(np.int64)))
    return (torch.stack(support).to(masks.device),
            torch.stack(comps).to(masks.device))


def symmetrized_affinity(logits: torch.Tensor,
                         support: torch.Tensor | None = None
                         ) -> torch.Tensor:
    """[B,8,H,W] symmetrized edge weights, optionally support-masked."""
    a = torch.sigmoid(logits)
    weights = []
    for index, (dy, dx) in enumerate(OFFSETS):
        reverse = a[:, REVERSE[index]]
        reverse_at_u = _shift_to(reverse, -dy, -dx)
        weight = 0.5 * (a[:, index] + reverse_at_u)
        if support is not None:
            support_v = _shift_to(support.float(), -dy, -dx)
            weight = weight * support.float() * support_v
        weights.append(weight)
    return torch.stack(weights, dim=1)


@torch.no_grad()
def build_dense_components(masks: torch.Tensor, stride: int) -> torch.Tensor:
    """Foreground component ids on the graph grid (0 = background)."""
    pooled = F.max_pool2d(masks[:, 0:1], stride) > 0.5
    labels = []
    for grid in pooled[:, 0].cpu().numpy():
        labels.append(torch.from_numpy(
            label(grid, connectivity=2).astype(np.int64)))
    return torch.stack(labels).to(masks.device)


def edge_loss(weights: torch.Tensor, comp: torch.Tensor) -> torch.Tensor:
    positives, negatives = [], []
    for index, (dy, dx) in enumerate(OFFSETS):
        comp_v = _shift_to(comp.float(), -dy, -dx).long()
        same = (comp > 0) & (comp_v > 0) & (comp == comp_v)
        diff = ~same
        prob = weights[index].clamp(1e-4, 1.0 - 1e-4)
        if same.any():
            positives.append(F.binary_cross_entropy(prob[same], torch.ones_like(prob[same])))
        if diff.any():
            negatives.append(F.binary_cross_entropy(prob[diff], torch.zeros_like(prob[diff])))
    if not positives or not negatives:
        return weights.sum() * 0.0
    return 0.5 * (torch.stack(positives).mean()
                  + torch.stack(negatives).mean())


@torch.no_grad()
def graph_distance(support: torch.Tensor, sources: np.ndarray,
                   steps: int) -> np.ndarray:
    """Unit-edge BFS distance from each source on the support graph."""
    s, height, width = len(sources), *support.shape
    dist = torch.full((s, height, width), 1e4, device=support.device)
    index = torch.arange(s, device=support.device)
    rows = torch.as_tensor(sources[:, 0], device=support.device)
    cols = torch.as_tensor(sources[:, 1], device=support.device)
    dist[index, rows, cols] = 0.0
    for _ in range(steps):
        new = dist
        for dy, dx in OFFSETS:
            # Distance arrays must not inherit the reachability zero-pad:
            # a zero outside the grid would act as a phantom source.
            new = torch.minimum(
                new, _shift_to(dist, dy, dx, pad=1e4) + 1.0)
        dist = torch.where(support.bool(), new,
                           torch.full_like(new, 1e4))
        dist[index, rows, cols] = 0.0
    return dist.cpu().numpy()


def reachability(weights: torch.Tensor, support: torch.Tensor,
                 sources: np.ndarray, steps: int) -> list[torch.Tensor]:
    """Max-Min DP for each source; returns r_t after every step."""
    s, height, width = len(sources), *support.shape
    r = torch.zeros((s, height, width), device=weights.device)
    index = torch.arange(s, device=weights.device)
    rows = torch.as_tensor(sources[:, 0], device=weights.device)
    cols = torch.as_tensor(sources[:, 1], device=weights.device)
    r[index, rows, cols] = 1.0
    history = []
    for _ in range(steps):
        new = r
        for axis, (dy, dx) in enumerate(OFFSETS):
            r_prev = _shift_to(r, dy, dx)
            w_prev = _shift_to(weights[axis], dy, dx)
            new = torch.maximum(new, torch.minimum(r_prev, w_prev))
        r = new * support.float()
        r[index, rows, cols] = 1.0
        history.append(r)
    return history


class WPRFLoss(nn.Module):
    def __init__(self, stride: int = 8, sources: int = 16,
                 scales: tuple[int, ...] = (1, 2, 4, 8, 16),
                 edge_weight: float = 1.0, reach_weight: float = 1.0):
        super().__init__()
        self.stride = int(stride)
        self.sources = int(sources)
        self.scales = tuple(int(k) for k in scales)
        self.edge_weight = float(edge_weight)
        self.reach_weight = float(reach_weight)

    def forward(self, affinity_head, embeddings: torch.Tensor,
                masks: torch.Tensor):
        logits = affinity_head(embeddings)
        support, comp = build_support(masks, self.stride)
        weights_all = symmetrized_affinity(logits)
        weights_support = symmetrized_affinity(logits, support)

        if self.edge_weight > 0:
            dense = build_dense_components(masks, self.stride)
            edge = torch.stack([
                edge_loss(weights_all[b], dense[b])
                for b in range(logits.shape[0])]).mean()
        else:
            edge = logits.sum() * 0.0
        if self.reach_weight > 0:
            reach = torch.stack([
                self._reach_loss(weights_support[b], support[b], comp[b])
                for b in range(logits.shape[0])]).mean()
        else:
            reach = logits.sum() * 0.0
        import os
        if os.environ.get('WPRF_DEBUG'):
            print(f'[wprf] edge={float(edge.detach()):.4f} '
                  f'reach={float(reach.detach()):.4f} '
                  f'nodes={int(support.sum())}', flush=True)
        return self.edge_weight * edge + self.reach_weight * reach

    def _reach_loss(self, weights: torch.Tensor, support: torch.Tensor,
                    comp: torch.Tensor) -> torch.Tensor:
        support_np = support.detach().cpu().numpy() > 0
        nodes = np.argwhere(support_np)
        if len(nodes) == 0:
            return weights.sum() * 0.0
        count = min(self.sources, len(nodes))
        rng = np.random.default_rng()
        sources = nodes[rng.choice(len(nodes), size=count, replace=False)]
        steps = max(self.scales)
        dist = graph_distance(support, sources, steps)
        comp_np = comp.detach().cpu().numpy()
        history = reachability(weights, support, sources, steps)

        losses = []
        for k in self.scales:
            low = int(math.ceil(k / 2.0))
            labels, values = [], []
            for s_index, source in enumerate(sources):
                source_comp = comp_np[source[0], source[1]]
                same = comp_np == source_comp
                d = dist[s_index]
                positive = same & (d >= low) & (d <= k) & (d < 1e4)
                negative = (((comp_np > 0) & (comp_np != source_comp))
                            | (same & (d > k)))
                for mask, label_value in ((positive, 1.0),
                                          (negative, 0.0)):
                    ys, xs = np.nonzero(mask)
                    if len(ys) == 0:
                        continue
                    j = rng.integers(len(ys))
                    values.append(history[k - 1][s_index, ys[j], xs[j]])
                    labels.append(label_value)
            if not values:
                continue
            values = torch.stack(values).clamp(1e-4, 1.0 - 1e-4)
            targets = torch.tensor(labels, device=values.device)
            losses.append(F.binary_cross_entropy(values, targets))
        if not losses:
            return weights.sum() * 0.0
        return torch.stack(losses).mean()
