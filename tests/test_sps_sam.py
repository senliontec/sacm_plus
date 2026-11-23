"""Fast unit tests for the SP-SAM package."""

import torch

from methods.sps_sam.model import PromptProposalHead, select_peaks
from methods.sps_sam.topology_losses import SoftclDiceLoss


def test_proposal_head_shape():
    head = PromptProposalHead(256)
    out = head(torch.zeros(2, 256, 64, 64))
    assert out.shape == (2, 1, 256, 256)


def test_select_peaks_respects_radius():
    heat = torch.zeros(64, 64)
    heat[10, 10] = 1.0
    heat[10, 11] = 0.9
    heat[50, 50] = 0.8
    peaks = select_peaks(heat, k=4, radius=2)
    assert peaks.shape == (2, 2)
    assert torch.allclose(peaks[0], torch.tensor([10.5, 10.5]))
    assert torch.allclose(peaks[1], torch.tensor([50.5, 50.5]))


def test_select_peaks_empty():
    peaks = select_peaks(torch.zeros(64, 64), k=4, radius=2)
    assert peaks.shape == (0, 2)


def test_soft_cldice_finite_and_differentiable():
    loss = SoftclDiceLoss(num_iter=3)
    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    target = (torch.rand(1, 1, 32, 32) > 0.95).float()
    value = loss(logits, target)
    value.backward()
    assert torch.isfinite(value)
    assert logits.grad is not None
