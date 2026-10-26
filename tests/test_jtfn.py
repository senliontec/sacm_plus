"""Fast JTFN adapter tests (no full-size training)."""

import random
from types import SimpleNamespace

import torch
from PIL import Image

from baseline_zoo.adapters.jtfn import JTFNAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.jtfn import build_jtfn


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
    assert 'jtfn' in available_adapters()
    adapter = load_adapter('jtfn')
    assert isinstance(adapter, JTFNAdapter)
    assert adapter.id == 'jtfn'
    assert adapter.input_size == 256
    assert adapter.official_epochs == 1000
    assert adapter.train_batch_size == 2
    assert any('scratch' in deviation for deviation in adapter.deviations)


def test_tiny_forward_and_boundary_loss():
    model = build_jtfn('base64', steps=3)
    images = torch.randn(1, 3, 64, 64)
    masks = torch.zeros(1, 1, 64, 64)
    masks[..., :32] = 1
    boundary = (masks - torch.nn.functional.max_pool2d(
        masks, 3, 1, 1)).abs()
    outputs = model({'img': images})
    assert outputs['output'].shape == (1, 1, 64, 64)
    loss = model.compute_objective(
        outputs, {'anno_mask': masks, 'anno_boundary': boundary})[
            'total_loss']
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_preprocess_shapes(tmp_path):
    adapter = JTFNAdapter()
    adapter.input_size = 256
    image, mask, boundary = adapter._preprocess(
        _sample(tmp_path), random.Random(0), train=True)
    assert image.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert boundary.shape == (1, 256, 256)
    assert set(mask.unique().tolist()) <= {0, 1}
    assert set(boundary.unique().tolist()) <= {0, 1}
    assert torch.isfinite(image).all()


def test_eval_padding(tmp_path):
    adapter = JTFNAdapter()
    adapter.input_size = 256
    prepared = adapter.prepare_eval_sample(
        _sample(tmp_path), torch.device('cpu'))
    height, width = prepared['original_size']
    padded_h, padded_w = prepared['image'].shape[-2:]
    assert padded_h % 32 == 0 and padded_w % 32 == 0
    assert padded_h >= height and padded_w >= width


def test_config_roundtrip():
    adapter = JTFNAdapter()
    adapter.input_size = 256
    args = SimpleNamespace(jtfn_backbone='base64', jtfn_steps=3)
    config = adapter.save_config(args)
    assert config['jtfn_backbone'] == 'base64'
    assert config['jtfn_steps'] == 3
    restore = SimpleNamespace()
    adapter.restore_config(restore, config)
    assert restore.input_size == 256
    assert restore.jtfn_backbone == 'base64'


def test_eval_downscales_oversized_image_and_restores_gt_shape(tmp_path):
    image = tmp_path / 'BIG_0_image.png'
    mask = tmp_path / 'BIG_0_mask.png'
    Image.new('RGB', (128, 96), (120, 120, 120)).save(image)
    label = Image.new('L', (128, 96), 0)
    label.save(mask)
    sample = Sample(image_path=image, mask_path=mask,
                    dataset='BIG', stem='BIG_0')
    adapter = JTFNAdapter()
    adapter.args = SimpleNamespace(jtfn_eval_max_side=64)
    prepared = adapter.prepare_eval_sample(sample, torch.device('cpu'))
    assert prepared['original_size'] == (96, 128)
    assert prepared['model_size'] == (48, 64)
    assert prepared['image'].shape[-2:] == (64, 64)

    class Dummy(torch.nn.Module):
        def forward(self, batch):
            h, w = batch['img'].shape[-2:]
            return {'output': torch.zeros(1, 1, h, w)}

    pred = adapter.predict_sample(Dummy(), prepared, sample)
    assert pred.shape == (96, 128)
