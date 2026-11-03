"""CAPE-style connectivity-aware path loss.

CAPE (Connectivity-Aware Path Enforcement) samples connected node pairs
from the ground-truth graph and penalizes the predicted cost along the
corresponding path.  The original formulation predicts a distance map and
uses Dijkstra; this implementation keeps the path-enforcement idea and
uses a deterministic BFS on the GT skeleton, which is cheap enough for a
training loop and still produces dense gradients along the whole path.

It is registered as ``cape`` so it can be selected with
``--topology_loss cape``.  This is a practical port, not a line-by-line
reproduction of the official implementation.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from skimage.morphology import skeletonize
except ImportError:  # pragma: no cover
    skeletonize = None

from .registry import register


_NEIGHBOURS = tuple(
    (dy, dx)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dy == 0 and dx == 0)
)


class CAPELoss(nn.Module):
    """Path-level connectivity loss on the GT skeleton.

    Args:
        resolution: work resolution (skeletonization is CPU-bound).
        num_pairs: number of connected endpoint pairs sampled per image.
        min_path: minimum path length in pixels.
        smooth: numerical stabilizer.
    """

    def __init__(
        self,
        resolution: int | None = 128,
        num_pairs: int = 4,
        min_path: int = 4,
        smooth: float = 1e-6,
        **_: object,
    ) -> None:
        super().__init__()
        if skeletonize is None:
            raise ImportError('CAPELoss requires scikit-image')
        self.resolution = resolution
        self.num_pairs = max(1, int(num_pairs))
        self.min_path = max(1, int(min_path))
        self.smooth = float(smooth)

    @staticmethod
    def _endpoints(coords: np.ndarray, coord_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
        endpoints = []
        for y, x in coords:
            degree = sum(
                (y + dy, x + dx) in coord_set for dy, dx in _NEIGHBOURS)
            if degree == 1:
                endpoints.append((int(y), int(x)))
        return endpoints

    @staticmethod
    def _bfs_path(
        coord_set: set[tuple[int, int]],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        if start == goal or start not in coord_set:
            return []
        queue = deque([start])
        parent = {start: None}
        while queue:
            node = queue.popleft()
            if node == goal:
                break
            y, x = node
            for dy, dx in _NEIGHBOURS:
                nxt = (y + dy, x + dx)
                if nxt in coord_set and nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)
        if goal not in parent:
            return []
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    def _image_paths(self, mask: np.ndarray) -> Iterable[list[tuple[int, int]]]:
        skeleton = skeletonize(mask)
        coords = np.argwhere(skeleton)
        if len(coords) < 2:
            return
        coord_set = {(int(y), int(x)) for y, x in coords}
        endpoints = self._endpoints(coords, coord_set)
        anchors = endpoints if len(endpoints) >= 2 else [
            tuple(coords[0]), tuple(coords[-1])]
        for index in range(self.num_pairs):
            start = anchors[index % len(anchors)]
            goal = anchors[(index + 1) % len(anchors)]
            path = self._bfs_path(coord_set, start, goal)
            if len(path) >= self.min_path:
                yield path

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or target.ndim != 4:
            raise ValueError('logits and target must be [B,1,H,W]')
        if self.resolution is not None and (
                logits.shape[-1] != self.resolution
                or logits.shape[-2] != self.resolution):
            logits = F.interpolate(
                logits, size=(self.resolution, self.resolution),
                mode='bilinear', align_corners=False)
            target = F.interpolate(
                target, size=(self.resolution, self.resolution),
                mode='bilinear', align_corners=False)
        prob = torch.sigmoid(logits)
        losses = []
        masks = target.detach().cpu().numpy() > 0.5
        for batch_index, mask in enumerate(masks):
            for path in self._image_paths(mask[0]):
                ys = torch.as_tensor(
                    [point[0] for point in path], device=logits.device,
                    dtype=torch.long)
                xs = torch.as_tensor(
                    [point[1] for point in path], device=logits.device,
                    dtype=torch.long)
                losses.append(1.0 - prob[batch_index, 0, ys, xs].mean())
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


register('cape')(CAPELoss)
