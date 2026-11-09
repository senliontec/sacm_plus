"""SAM-Med2D must start from the full SAM decoder when fine-tuning from SAM.

Regression for the official ``load_from`` behaviour that excluded
``mask_tokens``, ``output_hypernetworks_mlps`` and
``iou_prediction_head``: those tensors are shape-compatible with a
256-input SAM-Med2D model and must be inherited, not random-initialised.
"""

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.sam_med2d import SamMed2DAdapter
from baseline_zoo.data import Sample
from baseline_zoo.models.sam_med2d import build_sam_med2d
from baseline_zoo.models.sam_med2d.segment_anything.build_sam import (
    _checkpoint_coverage,
)

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


def test_empty_mask_uses_full_image_box():
    empty = torch.zeros(8, 16, dtype=torch.int64)
    boxes = SamMed2DAdapter._prompt_boxes(empty)
    assert boxes.shape == (1, 4)
    assert boxes.tolist() == [[0.0, 0.0, 15.0, 7.0]]


def test_empty_mask_train_and_eval_preparation(tmp_path):
    image_path = tmp_path / 'EMPTY_0.png'
    mask_path = tmp_path / 'EMPTY_0_mask.png'
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.zeros((12, 16), dtype=np.uint8)).save(mask_path)
    sample = Sample(image_path, mask_path, 'EMPTY', 'EMPTY_0')

    adapter = SamMed2DAdapter()
    adapter.args = SimpleNamespace(mask_num=1)

    prepared = adapter.prepare_eval_sample(sample, 'cpu')
    assert prepared['image'].shape == (1, 3, 256, 256)
    assert prepared['boxes'].shape == (1, 4)
    assert prepared['original_size'] == (12, 16)

    batch = adapter.prepare_train_batch([sample], random.Random(0))
    assert batch['image'].shape == (1, 3, 256, 256)
    assert batch['label'].shape == (1, 1, 256, 256)
    assert int(batch['label'].sum()) == 0
    assert batch['boxes'].shape == (1, 1, 4)
    assert batch['point_coords'].shape == (1, 1, 2)


def test_default_checkpoint_matches_model_type():
    assert SamMed2DAdapter._default_checkpoint('vit_b').name == (
        'sam-med2d_b.pth')
    assert SamMed2DAdapter._default_checkpoint('vit_l').name == (
        'sam_vit_l_0b3195.pth')
    assert SamMed2DAdapter._default_checkpoint('vit_h').name == (
        'sam_vit_h_4b8939.pth')


def test_checkpoint_coverage_allows_adapters_and_interpolated_positions():
    target = {
        'image_encoder.pos_embed': torch.zeros(1, 16, 16, 4),
        'image_encoder.blocks.0.Adapter.weight': torch.zeros(2),
        'mask_decoder.weight': torch.zeros(2),
    }
    state = {
        'image_encoder.pos_embed': torch.zeros(1, 64, 64, 4),
        'mask_decoder.weight': torch.zeros(2),
        'unexpected.weight': torch.zeros(2),
    }
    missing, unexpected, bad_missing = _checkpoint_coverage(target, state)
    assert missing == ['image_encoder.blocks.0.Adapter.weight']
    assert bad_missing == []
    assert unexpected == ['unexpected.weight']


def test_checkpoint_coverage_rejects_missing_encoder_tensor():
    target = {
        'image_encoder.blocks.0.Adapter.weight': torch.zeros(2),
        'image_encoder.blocks.0.attn.qkv.weight': torch.zeros(2),
    }
    state = {'image_encoder.blocks.0.Adapter.weight': torch.zeros(2)}
    _, _, bad_missing = _checkpoint_coverage(target, state)
    assert bad_missing == ['image_encoder.blocks.0.attn.qkv.weight']
