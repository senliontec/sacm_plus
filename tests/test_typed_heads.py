"""Smoke tests for the typed-head prototype and Tversky loss."""

import torch
from torch import nn

from core.losses.tversky import TverskyLoss
from models.sam_network.mask_decoder import MaskDecoder


class _DummyTransformer(nn.Module):
    """Return token outputs and a flattened source map; enough for decoder."""

    def forward(self, src, pos_src, tokens):
        b, c, h, w = src.shape
        return tokens, src.flatten(2).transpose(1, 2)


def _decoder(typed: bool) -> MaskDecoder:
    return MaskDecoder(
        transformer_dim=256,
        transformer=_DummyTransformer(),
        num_multimask_outputs=3,
        use_adapter_skip=False,
        use_coarse_to_fine=False,
        use_fusion_v2=False,
        use_multi_depth=False,
        use_typed_heads=typed,
    )


def test_typed_heads_have_stable_slots_and_gradients():
    decoder = _decoder(True)
    image = torch.randn(1, 256, 16, 16)
    pe = torch.zeros(1, 256, 16, 16)
    sparse = torch.zeros(1, 0, 256)
    dense = torch.zeros(1, 256, 16, 16)
    masks, iou, stage1 = decoder.predict_masks(
        image, pe, sparse, dense, return_stage1=True)

    assert masks.shape == (1, 4, 64, 64)
    assert iou.shape == (1, 4)
    assert stage1.shape == (1, 4, 64, 64)
    assert decoder.type_embeddings.shape == (4, 256)
    (masks.sum() + iou.sum() + stage1.sum()).backward()
    assert decoder.type_embeddings.grad is not None
    assert torch.count_nonzero(decoder.type_embeddings.grad) > 0


def test_untyped_decoder_has_no_type_embeddings():
    decoder = _decoder(False)
    assert not hasattr(decoder, 'type_embeddings')
    image = torch.randn(1, 256, 16, 16)
    pe = torch.zeros(1, 256, 16, 16)
    sparse = torch.zeros(1, 0, 256)
    dense = torch.zeros(1, 256, 16, 16)
    masks, iou = decoder.predict_masks(image, pe, sparse, dense)
    assert masks.shape == (1, 4, 64, 64)
    assert iou.shape == (1, 4)


def test_tversky_recall_loss_is_finite_and_differentiable():
    loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
    logits = torch.full((1, 1, 8, 8), -8.0, requires_grad=True)
    target = torch.ones(1, 1, 8, 8)
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) > 0
