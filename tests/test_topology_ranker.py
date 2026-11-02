"""Unit tests for the label-free topology ranker (no torch required)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from core.topology_ranker import (  # noqa: E402
    RankWeights,
    rank_candidates,
    skeleton_evidence,
    topology_descriptors,
    topology_stability,
    widest_path_bottleneck,
)


def _horizontal_line(
    shape=(64, 64), row=32, start=8, stop=56, width=1,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[row:row + width, start:stop] = True
    return mask


def test_topology_descriptors_single_line():
    mask = _horizontal_line(width=3)
    desc = topology_descriptors(mask)
    assert desc.b0 == 1
    assert desc.b1 == 0
    assert desc.euler == 1
    assert desc.skeleton_length > 0
    assert desc.frag_skel == pytest.approx(0.0, abs=1e-6)
    assert desc.largest_component_skeleton_share == pytest.approx(1.0)


def test_topology_descriptors_two_components():
    mask = _horizontal_line(row=20) | _horizontal_line(row=40)
    desc = topology_descriptors(mask)
    assert desc.b0 == 2
    assert desc.b1 == 0
    assert desc.frag_skel > 0.0
    assert desc.largest_component_skeleton_share < 1.0


def test_topology_descriptors_hole():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    mask[13:19, 13:19] = False
    desc = topology_descriptors(mask)
    assert desc.b0 == 1
    assert desc.b1 == 1
    assert desc.euler == 0


def test_topology_descriptors_empty_and_bad_shape():
    empty = topology_descriptors(np.zeros((16, 16), dtype=bool))
    assert empty.b0 == 0
    assert empty.frag_skel == pytest.approx(1.0)
    with pytest.raises(ValueError):
        topology_descriptors(np.zeros((2, 2, 2), dtype=bool))


def test_topology_stability_prefers_threshold_stable_candidate():
    stable = np.zeros((64, 64), dtype=np.float32)
    stable[20, 8:56] = 0.9
    stable[40, 8:56] = 0.9

    unstable = np.zeros((64, 64), dtype=np.float32)
    unstable[20, 8:56] = 0.9
    # This component disappears between threshold 0.4 and 0.5.
    unstable[40, 8:56] = 0.45

    stable_result = topology_stability(prob=stable)
    unstable_result = topology_stability(prob=unstable)
    assert stable_result.mode == 'probability'
    assert stable_result.score > unstable_result.score
    assert stable_result.b0_median == pytest.approx(2.0)
    assert unstable_result.b0_median < 2.0


def test_topology_stability_morphology_fallback():
    mask = _horizontal_line(width=3)
    result = topology_stability(mask=mask)
    assert result.mode == 'morphology'
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.frag_skel <= 1.0


def _bridge_setup(bridge_value: float):
    shape = (64, 64)
    mask = np.zeros(shape, dtype=bool)
    mask[32, 8:24] = True
    mask[32, 40:56] = True
    support = np.zeros(shape, dtype=np.float32)
    support[mask] = 1.0
    support[32, 24:40] = bridge_value
    return support, mask


def test_widest_path_prefers_high_support_bridge():
    high_support, mask = _bridge_setup(0.9)
    low_support, _ = _bridge_setup(0.05)
    high = widest_path_bottleneck(high_support, mask, stride=1)
    low = widest_path_bottleneck(low_support, mask, stride=1)
    assert high.components == 2
    assert high.score > low.score + 0.5
    assert high.score > 0.8
    assert low.score < 0.2


def test_widest_path_empty_and_single_component():
    empty = widest_path_bottleneck(
        np.zeros((16, 16), dtype=np.float32),
        np.zeros((16, 16), dtype=bool),
        stride=1,
    )
    assert empty.score == 0.0
    assert empty.components == 0

    single = widest_path_bottleneck(
        np.ones((16, 16), dtype=np.float32),
        np.ones((16, 16), dtype=bool),
        stride=1,
    )
    assert single.score == pytest.approx(1.0)
    assert single.components == 1


def test_rank_candidates_selects_clean_line():
    clean = _horizontal_line()
    fragmented = _horizontal_line(row=20) | _horizontal_line(row=40)
    empty = np.zeros((64, 64), dtype=bool)

    clean_prob = clean.astype(np.float32) * 0.9
    fragmented_prob = fragmented.astype(np.float32) * 0.9
    empty_prob = np.zeros((64, 64), dtype=np.float32)

    result = rank_candidates(
        [clean, fragmented, empty],
        probs=[clean_prob, fragmented_prob, empty_prob],
        weights=RankWeights(),
        stride=1,
    )
    assert result.selected == 0
    assert result.scores[0] > result.scores[1]
    assert result.scores[0] > result.scores[2]
    assert result.features[0]['s_wprf'] > result.features[1]['s_wprf']


def test_rank_candidates_empty_never_selected():
    clean = _horizontal_line()
    empty = np.zeros((64, 64), dtype=bool)
    clean_prob = clean.astype(np.float32) * 0.9
    empty_prob = np.zeros((64, 64), dtype=np.float32)

    result = rank_candidates(
        [empty, clean],
        probs=[empty_prob, clean_prob],
        stride=1,
    )
    assert result.selected == 1
    assert result.scores[1] > result.scores[0]


def test_rank_candidates_is_deterministic():
    clean = _horizontal_line()
    fragmented = _horizontal_line(row=20) | _horizontal_line(row=40)
    probs = [
        clean.astype(np.float32) * 0.9,
        fragmented.astype(np.float32) * 0.9,
    ]
    first = rank_candidates([clean, fragmented], probs=probs, stride=1)
    second = rank_candidates([clean, fragmented], probs=probs, stride=1)
    assert first.selected == second.selected
    assert first.scores == second.scores
    assert first.features == second.features


def test_rank_candidates_rejects_length_mismatch():
    mask = _horizontal_line()
    with pytest.raises(ValueError):
        rank_candidates([mask], probs=[])
    with pytest.raises(ValueError):
        rank_candidates([mask], iou_pred=[0.1, 0.2])


def test_skeleton_evidence_neutral_without_contrast():
    mask = _horizontal_line(width=3)
    image = np.full((64, 64), 0.5, dtype=np.float32)
    result = skeleton_evidence(image, mask)
    assert result.score == pytest.approx(0.5, abs=0.05)


def test_skeleton_evidence_shape_mismatch():
    with pytest.raises(ValueError):
        skeleton_evidence(
            np.zeros((32, 32), dtype=np.float32),
            np.zeros((64, 64), dtype=bool),
        )
