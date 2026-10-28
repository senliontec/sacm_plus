"""Skeleton/Morse-complex branch extraction with persistence pruning.

A2SAM's structural prompt generator converts the probabilistic U-Net
likelihood map into a Morse complex with discrete Morse theory and prunes
less salient branches with persistent homology.  The paper's exact cell
decomposition is not reproduced in the released code, so this module
implements the topology stage on the likelihood map's 8-connected skeleton:

* skeleton pixels are vertices, 8-neighbour links are edges;
* junction/endpoint pixels (degree != 2) are the critical cells of the
  skeleton graph and degree-2 chains are its 1-cells (branches);
* the geodesic distance from the highest-likelihood critical cell is the
  filtration function; a branch's persistence is the distance range it
  spans, and branches below an absolute or relative threshold are pruned
  (persistence-based pruning of the Morse complex);
* retained branches are sampled for the learnable branch gate in ``spg.py``.

Disconnected skeleton components are processed independently so a small
vessel is not discarded merely because it is disconnected from the largest
one.  This is a graph-based DMT/persistence approximation, documented as
such in ``UPSTREAM.md``; it is not a bit-exact reproduction of an unavailable
official implementation.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from skimage.morphology import skeletonize

_NEIGHBOURS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


@dataclass
class StructuralPoints:
    points: np.ndarray       # [N, 2] (x, y) in source pixels
    features: np.ndarray     # [N, 7]
    saliency: np.ndarray     # [N]
    n_critical: int
    n_branches: int

    @property
    def empty(self) -> bool:
        return self.points.shape[0] == 0


def _empty() -> StructuralPoints:
    return StructuralPoints(
        points=np.zeros((0, 2), np.float32),
        features=np.zeros((0, 7), np.float32),
        saliency=np.zeros((0,), np.float32),
        n_critical=0, n_branches=0)


def _neighbours(pixel, skeleton_set):
    y, x = pixel
    out = []
    for dy, dx in _NEIGHBOURS:
        candidate = (y + dy, x + dx)
        if candidate in skeleton_set:
            out.append(candidate)
    return out


def _components(pixels, skeleton_set) -> list[list[tuple[int, int]]]:
    seen = set()
    components = []
    for pixel in pixels:
        if pixel in seen:
            continue
        stack = [pixel]
        seen.add(pixel)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in _neighbours(node, skeleton_set):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def _dijkstra(root, neighbours):
    distances = {root: 0.0}
    heap = [(0.0, root)]
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distances.get(node, float('inf')):
            continue
        for neighbour in neighbours(node):
            step = 1.41421356237 if (
                neighbour[0] != node[0] and neighbour[1] != node[1]) else 1.0
            candidate = dist + step
            if candidate < distances.get(neighbour, float('inf')):
                distances[neighbour] = candidate
                heapq.heappush(heap, (candidate, neighbour))
    return distances


def _branches(component, skeleton_set, neighbours, distances):
    degrees = {pixel: len(neighbours(pixel)) for pixel in component}
    critical = {pixel for pixel, degree in degrees.items() if degree != 2}
    if not critical and component:
        # Pure cycle: promote one pixel to a critical cell.
        critical = {component[0]}
    paths = []
    seen = set()
    for start in critical:
        for first in neighbours(start):
            if tuple(sorted((start, first))) in seen:
                continue
            path = [start, first]
            seen.add(tuple(sorted((start, first))))
            previous, current = start, first
            while current not in critical:
                next_nodes = [n for n in neighbours(current)
                              if n != previous]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                seen.add(tuple(sorted((current, nxt))))
                path.append(nxt)
                previous, current = current, nxt
            if current in critical:
                paths.append(path)
    unique = {}
    for path in paths:
        key = (tuple(sorted((path[0], path[-1]))), len(path))
        if key not in unique:
            unique[key] = path
    return list(unique.values()), critical


def extract_structural_points(
    prob: np.ndarray,
    uncertainty: np.ndarray,
    *,
    persistence: float = 5.0,
    relative_persistence: float = 0.1,
    stride: int = 3,
    max_points: int = 512,
) -> StructuralPoints:
    """Extract pruned branch samples from a likelihood map."""
    prob = np.asarray(prob, dtype=np.float32)
    uncertainty = np.asarray(uncertainty, dtype=np.float32)
    if prob.ndim != 2 or uncertainty.shape != prob.shape:
        raise ValueError('prob and uncertainty must be matching 2-D maps')
    if stride < 1 or max_points < 1:
        raise ValueError('stride and max_points must be positive')

    mask = prob > 0.5
    if not mask.any():
        return _empty()
    skeleton = skeletonize(mask)
    skeleton_pixels = [tuple(p) for p in np.argwhere(skeleton)]
    if not skeleton_pixels:
        return _empty()
    skeleton_set = set(skeleton_pixels)
    components = _components(skeleton_pixels, skeleton_set)

    retained = []
    n_critical = 0
    for component in components:
        component_set = set(component)

        def component_neighbours(pixel, component_set=component_set):
            return _neighbours(pixel, component_set)

        root = max(component, key=lambda p: float(prob[p]))
        distances = _dijkstra(root, component_neighbours)
        paths, critical = _branches(
            component, set(component), component_neighbours, distances)
        n_critical += len(critical)
        stats = []
        for path in paths:
            distance = [distances.get(pixel, 0.0) for pixel in path]
            length = max(max(distance) - min(distance), 1.0)
            branch_prob = float(np.mean([prob[pixel] for pixel in path]))
            branch_unc = float(np.mean([uncertainty[pixel] for pixel in path]))
            stats.append({
                'path': path,
                'distance': distance,
                'length': length,
                'prob': branch_prob,
                'unc': branch_unc,
                'persistence': length,
            })
        if not stats:
            continue
        local_max = max(item['persistence'] for item in stats)
        threshold = max(float(persistence),
                        float(relative_persistence) * local_max)
        kept = [item for item in stats
                if item['persistence'] >= threshold]
        if not kept:
            kept = [max(stats, key=lambda item: item['persistence'])]
        retained.extend(kept)

    if not retained:
        return StructuralPoints(
            points=np.zeros((0, 2), np.float32),
            features=np.zeros((0, 7), np.float32),
            saliency=np.zeros((0,), np.float32),
            n_critical=n_critical, n_branches=0)

    max_persistence = max(item['persistence'] for item in retained) or 1.0
    max_dist = max(max(item['distance']) for item in retained) or 1.0
    max_length = max(item['length'] for item in retained) or 1.0
    max_saliency = max(
        item['persistence'] * item['prob'] * (1.0 + item['unc'])
        for item in retained) or 1.0

    points, features, saliency = [], [], []
    for item in retained:
        path = item['path']
        indices = list(range(0, len(path), stride))
        if indices[-1] != len(path) - 1:
            indices.append(len(path) - 1)
        branch_saliency = (item['persistence'] * item['prob']
                           * (1.0 + item['unc']))
        for index in indices:
            y, x = path[index]
            points.append([x, y])
            features.append([
                x / max(prob.shape[1] - 1, 1),
                y / max(prob.shape[0] - 1, 1),
                float(prob[y, x]),
                float(uncertainty[y, x]),
                item['persistence'] / max_persistence,
                item['distance'][index] / max_dist,
                item['length'] / max_length,
            ])
            saliency.append(branch_saliency / max_saliency)

    points = np.asarray(points, np.float32)
    features = np.asarray(features, np.float32)
    saliency = np.asarray(saliency, np.float32)
    if len(points) > max_points:
        order = np.argsort(-saliency)[:max_points]
        order.sort()
        points, features, saliency = (
            points[order], features[order], saliency[order])
    return StructuralPoints(
        points=points,
        features=features,
        saliency=saliency,
        n_critical=n_critical,
        n_branches=len(retained),
    )
