"""Unit tests for the candidate-level selection-diagnostic helpers."""

import importlib.util
import os

import numpy as np
import pytest
import torch

from core.evaluator import resolve_selection
from core.trainer import stage2_soft_weights

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'diagnose_selection.py')
_spec = importlib.util.spec_from_file_location('diagnose_selection', _SCRIPT)
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)


def test_best_index_and_regret():
    assert ds.best_index([0.1, 0.5, 0.3], higher=True) == 1
    assert ds.best_index([np.nan, 3.0, 1.0], higher=False) == 2
    assert ds.best_index([np.nan, np.nan], higher=True) == -1

    assert np.isclose(ds.regret_value(0.4, 0.6, higher=True), 0.2)
    assert np.isclose(ds.regret_value(3.0, 1.0, higher=False), 2.0)
    assert np.isnan(ds.regret_value(np.nan, 1.0, higher=True))


def test_spearman_and_ece():
    assert np.isclose(ds.safe_spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
    assert np.isclose(ds.safe_spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
    assert np.isclose(ds.safe_spearman([1, 2, np.nan, 4], [1, 2, 3, 4]), 1.0)
    assert np.isnan(ds.safe_spearman([1, 2, 3], [1, 1, 1]))

    assert np.isclose(ds.expected_calibration_error([0.0, 1.0], [0.0, 1.0]), 0.0)
    assert np.isclose(ds.expected_calibration_error([0.2, 0.8], [0.0, 1.0]), 0.2)


def test_usage_entropy():
    ent, counts = ds.usage_entropy([0, 0, 1, 1], 4)
    assert np.allclose(counts, [2, 2, 0, 0])
    assert np.isclose(ent, np.log(2.0))
    ent, counts = ds.usage_entropy([0, 0, 0, 0], 4)
    assert np.isclose(ent, 0.0)
    assert np.allclose(counts, [4, 0, 0, 0])


def test_candidate_metrics_identical_mask():
    gt = np.zeros((5, 5), dtype=np.float32)
    gt[2, 2] = 1.0
    logits = np.full((5, 5), -10.0, dtype=np.float32)
    logits[2, 2] = 10.0
    m = ds.compute_candidate_metrics(logits, gt, threshold=0.0,
                                     metrics=['dice', 'cldice', 'hd95'])
    assert np.isclose(m['dice'], 1.0)
    assert np.isclose(m['iou'], 1.0)
    assert np.isclose(m['cldice'], 1.0)
    assert np.isclose(m['hd95'], 0.0)


def test_name_path_and_category_parsing():
    assert ds.parse_name_path(['DRIVE:/data/drive']) == [('DRIVE', '/data/drive')]
    assert ds.parse_name_path(['/data/wire']) == [('wire', '/data/wire')]
    assert ds.parse_categories(['DRIVE:seen', 'WIRE:novel']) == {
        'DRIVE': 'seen', 'WIRE': 'novel'}


def test_pairwise_candidate_similarity():
    cands = np.zeros((2, 5, 5), dtype=np.float32)
    cands[0, 2, 2] = 10.0
    cands[1, 2, 2] = 10.0
    pair_iou, pair_cldice = ds.pairwise_candidate_similarity(cands, threshold=0.0)
    assert np.isclose(pair_iou, 1.0)
    assert np.isclose(pair_cldice, 1.0)


def test_stage2_soft_weights():
    iou = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    w = stage2_soft_weights(iou, mix=0.5, tau=1.0)
    assert w.shape == iou.shape
    assert torch.allclose(w.sum(dim=1), torch.ones(1))
    assert not w.requires_grad
    w_uniform = stage2_soft_weights(iou, mix=0.0)
    assert torch.allclose(w_uniform, torch.full_like(iou, 0.25))
    assert (w > 0).all()

    # D8 regression: every Stage-2 head must receive a nonzero gradient.
    m2 = torch.randn(1, 4, 3, 3, requires_grad=True)
    (m2 * w[:, :, None, None]).sum().backward()
    assert m2.grad is not None
    assert (m2.grad.abs() > 0).all()


def test_resolve_selection_never_uses_untrained_iou():
    assert resolve_selection('auto', 0.0) == 'gating'
    assert resolve_selection('auto', 1.0) == 'iou'
    assert resolve_selection('iou', 0.0) == 'iou'
    assert resolve_selection('gating', 1.0) == 'gating'
    with pytest.raises(ValueError):
        resolve_selection('bogus', 1.0)

