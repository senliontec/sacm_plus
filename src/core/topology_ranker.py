"""Label-free topology-aware ranking for typed Stage-2 candidates.

This module is intentionally pure NumPy/SciPy (no torch): it can be run
offline over candidate masks/probability maps saved by the evaluator, and it
is also safe to import from the regular evaluation path.

The two primary signals are:

* :func:`topology_stability` -- topology descriptors across probability
  thresholds (or small morphological perturbations when only binary masks
  are available), penalising candidates whose component/branch structure
  changes sharply.
* :func:`widest_path_bottleneck` -- a max-min (widest-path) score on a
  downsampled support graph, measuring whether disconnected components are
  connected by a high-confidence path.  This is the inference-time analogue
  of the WPRF widest-path idea.

Both are label-free.  GT is only used by the offline analysis script to
report oracle/selected metrics, never to choose the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage as ndi

from .metrics.common import skeletonize


_EIGHT = ndi.generate_binary_structure(2, 2)
_FOUR = ndi.generate_binary_structure(2, 1)
_NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


@dataclass(frozen=True)
class TopologyDescriptors:
    """Count-based topology descriptors for one binary candidate."""

    b0: int
    b1: int
    euler: int
    endpoints: int
    branchpoints: int
    skeleton_length: int
    largest_component_area_share: float
    largest_component_skeleton_share: float
    small_component_rate: float
    endpoint_density: float
    frag_skel: float

    def as_dict(self) -> dict[str, float]:
        return {
            'b0': self.b0,
            'b1': self.b1,
            'euler': self.euler,
            'endpoints': self.endpoints,
            'branchpoints': self.branchpoints,
            'skeleton_length': self.skeleton_length,
            'largest_component_area_share':
                self.largest_component_area_share,
            'largest_component_skeleton_share':
                self.largest_component_skeleton_share,
            'small_component_rate': self.small_component_rate,
            'endpoint_density': self.endpoint_density,
            'frag_skel': self.frag_skel,
        }


@dataclass(frozen=True)
class StabilityResult:
    """Result of a threshold/morphology topology-stability sweep."""

    score: float
    mode: str
    thresholds: tuple[float, ...]
    frag_skel: float
    b0_median: float
    descriptors: tuple[TopologyDescriptors, ...] = field(default=())


@dataclass(frozen=True)
class WidestPathResult:
    """Result of the widest-path bottleneck score."""

    score: float
    components: int
    largest_component_area: int
    bottleneck_by_component: tuple[float, ...] = field(default=())


@dataclass(frozen=True)
class EvidenceResult:
    """Skeleton-vs-ring image-evidence score."""

    score: float
    skeleton_mean: float
    ring_mean: float


@dataclass(frozen=True)
class RankWeights:
    """Fixed non-negative weights for the composite candidate score."""

    stability: float = 0.35
    widest_path: float = 0.35
    evidence: float = 0.10
    fragmentation: float = 0.15
    components: float = 0.05
    iou: float = 0.05

    def validate(self) -> None:
        values = (
            self.stability, self.widest_path, self.evidence,
            self.fragmentation, self.components, self.iou,
        )
        if any(float(v) < 0.0 for v in values):
            raise ValueError('RankWeights must be non-negative')
        if not any(float(v) > 0.0 for v in values):
            raise ValueError('at least one RankWeights term must be positive')


@dataclass(frozen=True)
class RankingResult:
    """Scores and the selected candidate index."""

    selected: int
    scores: tuple[float, ...]
    features: tuple[dict[str, object], ...]


def _empty_descriptors() -> TopologyDescriptors:
    return TopologyDescriptors(
        b0=0, b1=0, euler=0, endpoints=0, branchpoints=0,
        skeleton_length=0, largest_component_area_share=0.0,
        largest_component_skeleton_share=0.0,
        small_component_rate=1.0, endpoint_density=0.0, frag_skel=1.0,
    )


def topology_descriptors(
    mask: np.ndarray,
    *,
    small_area: int = 20,
) -> TopologyDescriptors:
    """Compute Betti/skeleton descriptors for a 2-D binary mask.

    ``b0`` uses 8-connectivity, ``b1`` counts background holes using
    4-connectivity (the standard pairing for a 2-D binary image).  An
    empty mask is assigned ``frag_skel=1`` and ``small_component_rate=1``
    so it can never win a fragmentation-aware ranking by accident.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f'mask must be 2-D, got shape {binary.shape}')
    if not binary.any():
        return _empty_descriptors()

    fg_labels, n_fg = ndi.label(binary, structure=_EIGHT)
    if n_fg == 0:
        return _empty_descriptors()

    areas = np.bincount(fg_labels.ravel())[1:]
    total_area = float(areas.sum())
    largest_area_share = float(areas.max() / total_area) if total_area else 0.0

    bg_labels, n_bg = ndi.label(~binary, structure=_FOUR)
    if n_bg:
        border = np.unique(np.concatenate([
            bg_labels[0, :], bg_labels[-1, :],
            bg_labels[:, 0], bg_labels[:, -1],
        ]))
        border = border[border != 0]
        b1 = int(max(n_bg - len(border), 0))
    else:
        b1 = 0

    skel = skeletonize(binary)
    skel_length = int(np.count_nonzero(skel))
    if skel_length:
        neighbor_count = (
            ndi.convolve(
                skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                mode='constant', cval=0,
            ).astype(np.int16) - skel.astype(np.int16)
        )
        endpoints = int(np.count_nonzero(skel & (neighbor_count == 1)))
        branchpoints = int(np.count_nonzero(skel & (neighbor_count >= 3)))
        # Skeleton pixels inherit their foreground component label.  This
        # avoids one skeletonization pass per component for fragmented masks.
        skel_labels = fg_labels[skel]
        skel_by_component = np.bincount(
            skel_labels, minlength=n_fg + 1)[1:]
        largest_skel_share = float(
            skel_by_component.max() / skel_length)
    else:
        endpoints = 0
        branchpoints = 0
        largest_skel_share = 0.0

    small_component_rate = float(np.mean(areas < int(small_area)))
    endpoint_density = (
        float(endpoints / skel_length) if skel_length else 0.0)

    return TopologyDescriptors(
        b0=int(n_fg),
        b1=b1,
        euler=int(n_fg - b1),
        endpoints=endpoints,
        branchpoints=branchpoints,
        skeleton_length=skel_length,
        largest_component_area_share=largest_area_share,
        largest_component_skeleton_share=largest_skel_share,
        small_component_rate=small_component_rate,
        endpoint_density=endpoint_density,
        frag_skel=float(np.clip(1.0 - largest_skel_share, 0.0, 1.0)),
    )


def _coefficient_of_variation(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    mean = float(arr.mean())
    if abs(mean) <= 1e-12:
        return 0.0 if float(arr.max()) == float(arr.min()) else 1.0
    return float(arr.std() / abs(mean))


def _stability_from_descriptors(
    descriptors: Sequence[TopologyDescriptors],
) -> tuple[float, float, float]:
    """Return (score, median frag_skel, median b0)."""
    if not descriptors:
        return 0.0, 1.0, 0.0
    if not any(d.skeleton_length > 0 for d in descriptors):
        return 0.0, 1.0, 0.0

    keys = (
        'b0', 'b1', 'skeleton_length', 'endpoints', 'branchpoints',
        'largest_component_skeleton_share', 'small_component_rate',
    )
    cvs = []
    for key in keys:
        values = [float(getattr(d, key)) for d in descriptors]
        cvs.append(_coefficient_of_variation(values))
    difficulty = float(np.mean(cvs)) if cvs else 0.0
    score = 1.0 / (1.0 + difficulty)
    frag = float(np.median([d.frag_skel for d in descriptors]))
    b0 = float(np.median([d.b0 for d in descriptors]))
    return float(np.clip(score, 0.0, 1.0)), frag, b0


def topology_stability(
    prob: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    *,
    thresholds: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    perturbation_radii: Iterable[int] = (0, 1, 2),
) -> StabilityResult:
    """Measure topology stability across thresholds or morphology.

    If ``prob`` is given, the sweep uses probability thresholds and is the
    intended MTTS score.  If only ``mask`` is available, the sweep applies
    small erosion/dilation perturbations; this is a weaker proxy and is
    reported as ``mode='morphology'``.
    """
    if prob is not None:
        probs = np.asarray(prob, dtype=np.float32)
        if probs.ndim != 2:
            raise ValueError(f'prob must be 2-D, got shape {probs.shape}')
        probs = np.clip(probs, 0.0, 1.0)
        thresholds = tuple(float(t) for t in thresholds)
        variants = [probs >= t for t in thresholds]
        mode = 'probability'
        used_thresholds = thresholds
    else:
        if mask is None:
            raise ValueError('topology_stability needs prob or mask')
        binary = np.asarray(mask, dtype=bool)
        if binary.ndim != 2:
            raise ValueError(f'mask must be 2-D, got shape {binary.shape}')
        radii = tuple(int(r) for r in perturbation_radii)
        variants = []
        for radius in radii:
            if radius == 0:
                variants.append(binary)
            else:
                variants.append(ndi.binary_erosion(
                    binary, structure=_EIGHT, iterations=radius))
                variants.append(ndi.binary_dilation(
                    binary, structure=_EIGHT, iterations=radius))
        mode = 'morphology'
        used_thresholds = tuple(float(r) for r in radii)

    descriptors = [topology_descriptors(v) for v in variants]
    score, frag, b0 = _stability_from_descriptors(descriptors)
    return StabilityResult(
        score=score,
        mode=mode,
        thresholds=used_thresholds,
        frag_skel=frag,
        b0_median=b0,
        descriptors=tuple(descriptors),
    )


def _block_max(array: np.ndarray, stride: int) -> np.ndarray:
    """Max-pool by an integer stride, padding the bottom/right with zeros."""
    if stride <= 1:
        return np.asarray(array)
    height, width = array.shape
    pad_h = (-height) % stride
    pad_w = (-width) % stride
    if pad_h or pad_w:
        array = np.pad(
            array, ((0, pad_h), (0, pad_w)), mode='constant',
            constant_values=0)
    pooled = array.reshape(
        array.shape[0] // stride, stride,
        array.shape[1] // stride, stride,
    ).max(axis=(1, 3))
    return pooled


def _crop_bbox(
    mask: np.ndarray,
    margin: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return mask, (0, 0, mask.shape[0], mask.shape[1])
    height, width = mask.shape
    y0 = max(0, int(ys.min()) - int(margin))
    y1 = min(height, int(ys.max()) + int(margin) + 1)
    x0 = max(0, int(xs.min()) - int(margin))
    x1 = min(width, int(xs.max()) + int(margin) + 1)
    return mask[y0:y1, x0:x1], (y0, y1, x0, x1)


def widest_path_bottleneck(
    support: np.ndarray,
    mask: np.ndarray,
    *,
    stride: int = 4,
    margin: int = 16,
) -> WidestPathResult:
    """Multi-source max-min widest-path score on a downsampled grid.

    ``support`` is a probability/support map in ``[0, 1]``.  The largest
    foreground component is used as a multi-source seed; every other
    component receives the maximum bottleneck (minimum support along the
    best path) from that source set.  A high score means the fragments are
    connected by high-confidence paths, i.e. they are plausible repairs
    rather than true separations.

    Only the mask bounding box (plus ``margin``) is processed.  The graph
    is stride-pooled, so the default cost is ~65k nodes at 1024x1024.
    """
    support = np.asarray(support, dtype=np.float32)
    binary = np.asarray(mask, dtype=bool)
    if support.shape != binary.shape:
        raise ValueError(
            f'support/mask shape mismatch: {support.shape} vs {binary.shape}')
    if support.ndim != 2:
        raise ValueError(f'support must be 2-D, got shape {support.shape}')
    if not binary.any():
        return WidestPathResult(score=0.0, components=0,
                                largest_component_area=0,
                                bottleneck_by_component=())

    cropped_mask, (y0, y1, x0, x1) = _crop_bbox(binary, margin)
    cropped_support = support[y0:y1, x0:x1]

    pooled_mask = _block_max(
        cropped_mask.astype(np.float32), stride) > 0.5
    pooled_support = _block_max(cropped_support, stride)
    if not pooled_mask.any():
        return WidestPathResult(score=0.0, components=0,
                                largest_component_area=0,
                                bottleneck_by_component=())

    labels, n_components = ndi.label(pooled_mask, structure=_EIGHT)
    if n_components == 0:
        return WidestPathResult(score=0.0, components=0,
                                largest_component_area=0,
                                bottleneck_by_component=())
    areas = np.bincount(labels.ravel())[1:]
    largest_label = int(np.argmax(areas)) + 1
    largest_area = int(areas[largest_label - 1])
    if n_components == 1:
        return WidestPathResult(score=1.0, components=1,
                                largest_component_area=largest_area,
                                bottleneck_by_component=(1.0,))

    height, width = pooled_mask.shape
    size = height * width
    best = np.full(size, -1.0, dtype=np.float32)
    heap: list[tuple[float, int]] = []

    for flat_index in np.flatnonzero(labels.ravel() == largest_label):
        idx = int(flat_index)
        best[idx] = 1.0
        heapq.heappush(heap, (-1.0, idx))

    support_flat = pooled_support.ravel()
    while heap:
        neg_bottleneck, current = heapq.heappop(heap)
        bottleneck = -neg_bottleneck
        if bottleneck < float(best[current]) - 1e-7:
            continue
        current_y, current_x = divmod(current, width)
        for dy, dx in _NEIGHBOR_OFFSETS:
            ny, nx = current_y + dy, current_x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            neighbor = ny * width + nx
            edge = min(float(support_flat[current]),
                       float(support_flat[neighbor]))
            candidate = min(bottleneck, edge)
            if candidate > float(best[neighbor]) + 1e-7:
                best[neighbor] = candidate
                heapq.heappush(heap, (-candidate, neighbor))

    component_bottlenecks: list[float] = []
    component_areas: list[int] = []
    flat_labels = labels.ravel()
    for label in range(1, n_components + 1):
        if label == largest_label:
            continue
        pixels = np.flatnonzero(flat_labels == label)
        if pixels.size == 0:
            continue
        component_bottlenecks.append(float(best[pixels].max()))
        component_areas.append(int(pixels.size))

    if not component_bottlenecks:
        return WidestPathResult(score=1.0, components=1,
                                largest_component_area=largest_area,
                                bottleneck_by_component=(1.0,))

    weights = np.asarray(component_areas, dtype=np.float64)
    values = np.asarray(component_bottlenecks, dtype=np.float64)
    weighted = float(np.sum(values * weights) / np.sum(weights))
    worst = float(values.min())
    score = float(np.clip(0.7 * weighted + 0.3 * worst, 0.0, 1.0))
    return WidestPathResult(
        score=score,
        components=int(n_components),
        largest_component_area=largest_area,
        bottleneck_by_component=tuple(component_bottlenecks),
    )


def _ridge_response(image: np.ndarray) -> np.ndarray:
    """Best-effort ridge/vesselness response, per-image normalised."""
    try:
        from skimage.filters import sato
        response = sato(
            image, sigmas=(1.0, 2.0), black_ridges=False)
    except Exception:
        # Fallback: gradient magnitude.  Weaker for centrelines, but keeps
        # the ranker usable when scikit-image is unavailable.
        response = ndi.gaussian_gradient_magnitude(image, sigma=1.5)
    response = np.asarray(response, dtype=np.float32)
    response -= float(response.min())
    peak = float(response.max())
    if peak > 1e-12:
        response /= peak
    else:
        response.fill(0.0)
    return response


def skeleton_evidence(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    ring_width: int = 3,
) -> EvidenceResult:
    """Compare image ridge response on the skeleton vs a surrounding ring.

    The score is in ``[0, 1]``; ``0.5`` is neutral (no contrast).  This is a
    weak feature by design: it is useful for vessels/roads/cracks but can be
    unreliable for low-contrast or non-ridge modalities.
    """
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    if gray.ndim != 2:
        raise ValueError(f'image must be 2-D or HxWxC, got {gray.shape}')
    if gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255.0
    else:
        gray = gray.astype(np.float32)
        if gray.max() > 1.0:
            gray /= 255.0

    binary = np.asarray(mask, dtype=bool)
    if binary.shape != gray.shape:
        raise ValueError(
            f'image/mask shape mismatch: {gray.shape} vs {binary.shape}')
    skel = skeletonize(binary)
    if not skel.any():
        return EvidenceResult(score=0.0, skeleton_mean=0.0, ring_mean=0.0)

    response = _ridge_response(gray)
    ring = ndi.binary_dilation(
        skel, structure=_EIGHT, iterations=int(ring_width)) & ~skel
    if not ring.any():
        ring = ndi.binary_dilation(skel, structure=_EIGHT)
    skel_mean = float(response[skel].mean())
    ring_mean = float(response[ring].mean()) if ring.any() else skel_mean
    score = float(np.clip((skel_mean - ring_mean + 1.0) / 2.0, 0.0, 1.0))
    return EvidenceResult(
        score=score, skeleton_mean=skel_mean, ring_mean=ring_mean)


def rank_candidates(
    candidates: Sequence[np.ndarray],
    *,
    probs: Sequence[np.ndarray] | None = None,
    image: np.ndarray | None = None,
    iou_pred: Sequence[float] | None = None,
    weights: RankWeights | None = None,
    thresholds: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    stride: int = 4,
) -> RankingResult:
    """Rank binary candidates with label-free topology features.

    ``probs`` is optional.  When absent, :func:`topology_stability` uses
    morphological perturbations and :func:`widest_path_bottleneck` uses a
    Gaussian-smoothed mask as its support map.  The ranking is deterministic
    and uses the first candidate as the tie-breaker.
    """
    candidates = [np.asarray(c, dtype=bool) for c in candidates]
    if not candidates:
        raise ValueError('rank_candidates needs at least one candidate')
    if probs is not None and len(probs) != len(candidates):
        raise ValueError('probs and candidates must have equal length')
    if iou_pred is not None and len(iou_pred) != len(candidates):
        raise ValueError('iou_pred and candidates must have equal length')
    weights = weights or RankWeights()
    weights.validate()

    descriptors = [topology_descriptors(c) for c in candidates]
    max_log_components = max(
        (np.log1p(max(d.b0, 0)) for d in descriptors), default=1.0)
    if max_log_components <= 0.0:
        max_log_components = 1.0

    features: list[dict[str, object]] = []
    raw_scores: list[float] = []
    for index, candidate in enumerate(candidates):
        prob = None if probs is None else np.asarray(
            probs[index], dtype=np.float32)
        stability = topology_stability(
            prob=prob, mask=candidate, thresholds=thresholds)
        support = (
            np.clip(prob, 0.0, 1.0) if prob is not None
            else ndi.gaussian_filter(
                candidate.astype(np.float32), sigma=2.0))
        widest = widest_path_bottleneck(
            support, candidate, stride=stride)
        evidence = (
            skeleton_evidence(image, candidate)
            if image is not None
            else EvidenceResult(score=0.5, skeleton_mean=0.0, ring_mean=0.0))
        iou_score = (
            float(iou_pred[index]) if iou_pred is not None else 0.0)
        desc = descriptors[index]

        score = (
            weights.stability * stability.score
            + weights.widest_path * widest.score
            + weights.evidence * evidence.score
            - weights.fragmentation * desc.frag_skel
            - weights.components * (
                np.log1p(max(desc.b0, 0)) / max_log_components)
            + weights.iou * iou_score
        )
        raw_scores.append(float(score))
        features.append({
            'score': float(score),
            's_stab': float(stability.score),
            's_wprf': float(widest.score),
            's_ev': float(evidence.score),
            'stab_mode': stability.mode,
            'frag_skel': float(desc.frag_skel),
            'b0': int(desc.b0),
            'b1': int(desc.b1),
            'endpoints': int(desc.endpoints),
            'branchpoints': int(desc.branchpoints),
            'skeleton_length': int(desc.skeleton_length),
            'largest_component_skeleton_share':
                float(desc.largest_component_skeleton_share),
            'small_component_rate': float(desc.small_component_rate),
            'wprf_components': int(widest.components),
            'iou_head_score': float(iou_score),
        })

    selected = int(np.argmax(np.asarray(raw_scores, dtype=np.float64)))
    return RankingResult(
        selected=selected,
        scores=tuple(raw_scores),
        features=tuple(features),
    )


__all__ = [
    'EvidenceResult',
    'RankWeights',
    'RankingResult',
    'StabilityResult',
    'TopologyDescriptors',
    'WidestPathResult',
    'rank_candidates',
    'skeleton_evidence',
    'topology_descriptors',
    'topology_stability',
    'widest_path_bottleneck',
]
