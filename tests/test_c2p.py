"""Fast C2P adapter tests (random init only)."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.c2p import C2PAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.c2p import build_c2p


def _sample(tmp_path):
    image = tmp_path / 'TEST_0_image.png'
    mask = tmp_path / 'TEST_0_mask.png'
    Image.new('RGB', (64, 48), (110, 100, 90)).save(image)
    label = Image.new('L', (64, 48), 0)
    for x in range(8, 56):
        label.putpixel((x, 24), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem='TEST_0')


def test_registry_and_metadata():
    assert 'c2p' in available_adapters()
    adapter = load_adapter('c2p')
    assert isinstance(adapter, C2PAdapter)
    assert adapter.id == 'c2p'
    assert adapter.input_size == 384
    assert adapter.training_free
    assert any('zero-shot' in item for item in adapter.deviations)


def test_random_init_forward_shapes():
    model = build_c2p(random_init=True).eval()
    with torch.no_grad():
        logits, logits_bg, geo, sem = model(torch.zeros(1, 3, 384, 384))
    assert logits.shape == (1, 1, 384, 384)
    assert logits_bg.shape == (1, 1, 384, 384)
    assert set(geo) == {
        'bbox', 'area', 'perimeter', 'aspect_ratio', 'compactness',
        'centroid', 'eccentricity', 'orientation', 'solidity'}
    assert geo['centroid'].shape == (1, 2)
    assert sem.shape == (1, 9, 768)


def test_mask_properties():
    mask = torch.zeros(10, 10)
    mask[2:6, 3:7] = 1
    area, centroid = C2PAdapter._mask_props(mask)
    assert area == pytest.approx(0.16)
    assert centroid[0].item() == pytest.approx((3 + 4 + 5 + 6) / 4 / 10)
    assert centroid[1].item() == pytest.approx((2 + 3 + 4 + 5) / 4 / 10)


def test_pair_is_imagenet_normalized(tmp_path):
    adapter = C2PAdapter()
    adapter.input_size = 384
    adapter.args = SimpleNamespace(c2p_tta=1)
    prepared = adapter.prepare_eval_sample(_sample(tmp_path), 'cpu')
    assert prepared['image'].shape == (1, 3, 384, 384)
    assert prepared['image'].mean().abs() < 1.0
    assert prepared['size'] == (64, 48)


def test_tta_prediction_shape_and_values(tmp_path):
    adapter = C2PAdapter()
    adapter.input_size = 384
    adapter.args = SimpleNamespace(c2p_tta=1)

    class _Fake(torch.nn.Module):
        def forward(self, image):
            logits = torch.full((image.shape[0], 1, 384, 384), -5.0)
            logits[:, :, 100:200, :] = 5.0
            geo = {
                'area': torch.full((image.shape[0], 1), 0.1),
                'centroid': torch.full((image.shape[0], 2), 0.5),
            }
            return logits, logits, geo, None

    prepared = adapter.prepare_eval_sample(_sample(tmp_path), 'cpu')
    pred = adapter.predict_sample(_Fake(), prepared, _sample(tmp_path))
    assert pred.shape == (48, 64)
    assert pred.dtype == bool


def test_config_roundtrip():
    adapter = C2PAdapter()
    args = SimpleNamespace(
        input_size=384,
        c2p_checkpoint='checkpoints/c2p/1.pth',
        c2p_random_init=False,
        c2p_tta=1,
        data_root='data/sacm_3shot_seed0',
    )
    config = adapter.save_config(args)
    restored = C2PAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 384
    assert target.c2p_tta == 1
    assert str(target.c2p_checkpoint).endswith('1.pth')


@pytest.mark.slow
def test_real_c2p_checkpoint_if_present():
    from pathlib import Path
    checkpoint = Path('checkpoints/c2p/1.pth')
    if not checkpoint.is_file():
        pytest.skip('missing official C2P checkpoint')
    model = build_c2p(checkpoint)
    with torch.no_grad():
        logits, _, _, _ = model(torch.zeros(1, 3, 384, 384))
    assert logits.shape == (1, 1, 384, 384)
