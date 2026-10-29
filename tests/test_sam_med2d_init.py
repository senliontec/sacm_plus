"""SAM-Med2D must start from the full SAM decoder when fine-tuning from SAM.

Regression for the official ``load_from`` behaviour that excluded
``mask_tokens``, ``output_hypernetworks_mlps`` and
``iou_prediction_head``: those tensors are shape-compatible with a
256-input SAM-Med2D model and must be inherited, not random-initialised.
"""

from pathlib import Path

import pytest
import torch

from baseline_zoo.adapters.sam_med2d import SamMed2DAdapter
from baseline_zoo.models.sam_med2d import build_sam_med2d

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth'


def test_postprocess_tracks_configured_input_size():
    adapter = SamMed2DAdapter()
    adapter.input_size = 512
    low_res = torch.zeros(1, 1, 64, 64)
    out = adapter._postprocess(low_res, (300, 400))
    assert out.shape == (1, 1, 300, 400)
    out = adapter._postprocess(low_res, (200, 300))
    assert out.shape == (1, 1, 200, 300)


def test_official_vit_b_checkpoint_roundtrip(tmp_path):
    # Mirrors the official sam-med2d_b.pth layout: a 256-input ViT-B model
    # saved under a top-level ``model`` key.
    source = build_sam_med2d(
        None, model_type='vit_b', image_size=256, encoder_adapter=True)
    checkpoint = tmp_path / 'sam-med2d_b.pth'
    torch.save({'model': source.state_dict()}, checkpoint)
    restored = build_sam_med2d(
        checkpoint, model_type='vit_b', image_size=256,
        encoder_adapter=True)
    for key, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value), key


@pytest.mark.skipif(not CHECKPOINT.is_file(),
                    reason='SAM ViT-B checkpoint is not present')
def test_sam_med2d_inherits_full_sam_decoder():
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model = build_sam_med2d(
        CHECKPOINT, model_type='vit_b', image_size=256,
        encoder_adapter=True)
    target = model.state_dict()

    decoder_keys = [
        key for key in state
        if any(name in key for name in (
            'mask_tokens', 'output_hypernetworks_mlps',
            'iou_prediction_head'))]
    assert decoder_keys
    for key in decoder_keys:
        assert torch.equal(target[key], state[key]), key

    # Image-size-dependent embeddings still need interpolation, not
    # verbatim loading.
    assert target['image_encoder.pos_embed'].shape == (1, 16, 16, 768)
    assert state['image_encoder.pos_embed'].shape == (1, 64, 64, 768)

    # Adapters have no SAM checkpoint tensors; they remain trainable/random.
    assert any('Adapter' in key for key in target)
