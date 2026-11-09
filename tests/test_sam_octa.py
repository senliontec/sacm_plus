"""SAM-OCTA backbone/config regression tests (no heavy forward)."""

import random
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from baseline_zoo.adapters.sam_octa import SAMOCTAAdapter
from baseline_zoo.data import Sample


def test_checkpoint_matches_model_type():
    adapter = SAMOCTAAdapter()
    assert adapter._default_checkpoint('vit_b').name == 'sam_vit_b_01ec64.pth'
    assert adapter._default_checkpoint('vit_l').name == 'sam_vit_l_0b3195.pth'
    assert adapter._default_checkpoint('vit_h').name == 'sam_vit_h_4b8939.pth'


def test_input_size_matches_official_backbone():
    assert SAMOCTAAdapter._input_size_for_model('vit_b') == 224
    assert SAMOCTAAdapter._input_size_for_model('vit_l') == 1024
    assert SAMOCTAAdapter._input_size_for_model('vit_h') == 1024


def test_config_roundtrip_keeps_backbone(tmp_path):
    adapter = SAMOCTAAdapter()
    args = SimpleNamespace(
        input_size=1024, sam_octa_model_type='vit_h',
        sam_checkpoint=tmp_path / 'missing.pth',
        sam_octa_threshold=0.7)
    config = adapter.save_config(args)
    assert config['model_type'] == 'vit_h'
    assert config['sam_octa_threshold'] == 0.7
    restored = SimpleNamespace()
    adapter.restore_config(restored, config)
    assert restored.sam_octa_model_type == 'vit_h'
    assert restored.input_size == 1024
    assert restored.sam_octa_threshold == 0.7


def _write_sample(tmp_path, mask_values):
    image_path = tmp_path / 'IMG_0.png'
    mask_path = tmp_path / 'IMG_0_mask.png'
    image = np.zeros((32, 24, 3), dtype=np.uint8)
    image[..., 1] = 80
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask_values.astype(np.uint8)).save(mask_path)
    return Sample(image_path, mask_path, 'IMG', 'IMG_0')


def test_literal_binary_mask_is_loaded_as_foreground(tmp_path):
    mask = np.zeros((32, 24), dtype=np.uint8)
    mask[4:10, 5:12] = 1
    sample = _write_sample(tmp_path, mask)
    label = SAMOCTAAdapter()._load_label(sample)
    assert label.dtype == np.uint8
    assert int(label.sum()) == 42


def test_validation_prompts_are_reseeded_per_pass(tmp_path):
    mask = np.zeros((32, 24), dtype=np.uint8)
    mask[4:10, 5:12] = 255
    mask[18:24, 14:20] = 255
    sample = _write_sample(tmp_path, mask)

    adapter = SAMOCTAAdapter()
    adapter.args = SimpleNamespace(seed=7, stage='train')
    adapter.device = torch.device('cpu')
    adapter.begin_validation()
    first = adapter.prepare_eval_sample(sample, adapter.device)

    random.seed(999)
    np.random.seed(999)
    adapter.begin_validation()
    second = adapter.prepare_eval_sample(sample, adapter.device)

    assert torch.equal(first['point_coords'], second['point_coords'])
    assert torch.equal(first['point_labels'], second['point_labels'])
