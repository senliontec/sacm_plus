"""Fast Matcher adapter tests (no DINOv2/SAM build)."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from baseline_zoo.adapters.matcher import MatcherAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.runtime import run_training_free


def _sample(tmp_path):
    image = tmp_path / 'TEST_0_image.png'
    mask = tmp_path / 'TEST_0_mask.png'
    Image.new('RGB', (32, 24), (120, 120, 120)).save(image)
    label = Image.new('L', (32, 24), 0)
    for x in range(16):
        for y in range(24):
            label.putpixel((x, y), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem='TEST_0')


def test_registry_and_metadata():
    assert 'matcher' in available_adapters()
    adapter = load_adapter('matcher')
    assert isinstance(adapter, MatcherAdapter)
    assert adapter.id == 'matcher'
    assert adapter.training_free is True
    assert adapter.input_size == 518
    assert adapter.official_epochs == 0


def test_support_and_target_shapes(tmp_path):
    adapter = MatcherAdapter()
    adapter.args = SimpleNamespace(
        support_shots=3, support_dataset='auto')
    adapter._support_root = Path('data/sacm_3shot_seed0')
    images, masks = adapter._support_tensors('DRIVE', torch.device('cpu'))
    assert images.shape == (1, 3, 3, 518, 518)
    assert masks.shape == (1, 3, 518, 518)

    prepared = adapter.prepare_eval_sample(
        _sample(tmp_path), torch.device('cpu'))
    assert prepared['image'].shape == (1, 3, 518, 518)
    assert prepared['original_size'] == (24, 32)


def test_training_free_checkpoint(tmp_path):
    adapter = MatcherAdapter()
    root = Path(tempfile.mkdtemp(prefix='matcher_tf_'))
    args = SimpleNamespace(
        data_root=Path('data/sacm_3shot_seed0'), workdir=root,
        seed=0, device='cpu', batch_size=1, limit_train=0, limit_val=0,
        selection='dice', resume=False,
        sam_weights=Path('checkpoints/sam_vit_h_4b8939.pth'),
        dinov2_weights=Path('checkpoints/dinov2_vitl14_pretrain.pth'),
        matcher_random_init=True, support_shots=3,
        support_dataset='auto')
    ckpt = run_training_free(adapter, args)
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert payload['adapter'] == 'matcher'
    assert payload['model_state_dict'] == {}
    assert payload['config']['training_free'] is True
    assert payload['input_size'] == 518


def test_config_roundtrip(tmp_path):
    adapter = MatcherAdapter()
    args = SimpleNamespace(
        sam_weights=tmp_path / 'missing_sam.pth',
        dinov2_weights=tmp_path / 'missing_dino.pth',
        matcher_random_init=True, support_shots=3,
        support_dataset='auto', data_root=Path('data/sacm_3shot_seed0'))
    config = adapter.save_config(args)
    assert config['support_shots'] == 3
    assert config['matcher_random_init'] is True
    restore = SimpleNamespace()
    adapter.restore_config(restore, config)
    assert restore.support_shots == 3
    assert restore.matcher_random_init is True
    assert restore.input_size == 518
