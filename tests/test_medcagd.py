"""Fast MedCAGD adapter tests (no full-size training)."""

import random
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.medcagd import MedCAGDAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.medcagd.losses import (
    MedCAGDLoss,
    generate_edge_targets,
)


def _sample(tmp_path):
    image = tmp_path / 'TEST_0_image.png'
    mask = tmp_path / 'TEST_0_mask.png'
    Image.new('RGB', (40, 32), (120, 120, 120)).save(image)
    label = Image.new('L', (40, 32), 0)
    for x in range(8, 28):
        label.putpixel((x, 16), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem='TEST_0')


def test_registry_and_metadata():
    assert 'medcagd' in available_adapters()
    adapter = load_adapter('medcagd')
    assert isinstance(adapter, MedCAGDAdapter)
    assert adapter.id == 'medcagd'
    assert adapter.input_size == 256
    assert adapter.train_batch_size == 8
    assert 'MedCAGD' in adapter.official_source
    assert any('256' in item for item in adapter.deviations)


def test_edge_targets_and_loss():
    masks = torch.zeros(1, 1, 16, 16)
    masks[:, :, 4:12, 4:12] = 1
    edges = generate_edge_targets(masks, radius=1)
    assert edges.shape == masks.shape
    assert edges[:, :, 4:12, 4:12].min() == 0
    assert edges.sum() > 0
    loss = MedCAGDLoss()(
        torch.zeros(1, 1, 16, 16, requires_grad=True),
        [torch.zeros(1, 1, 16, 16, requires_grad=True)] * 3,
        [torch.zeros(1, 1, 16, 16, requires_grad=True)] * 3,
        masks,
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_train_batch_and_tiling(tmp_path):
    adapter = MedCAGDAdapter()
    adapter.input_size = 16
    image, mask = adapter._train_pair(_sample(tmp_path), random.Random(0))
    assert image.shape == (3, 16, 16)
    assert mask.shape == (1, 16, 16)
    assert image.min() >= 0 and image.max() <= 1
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert adapter._tile_positions(16, 16, 8) == [0]
    assert adapter._tile_positions(40, 16, 8) == [0, 8, 16, 24]
    assert adapter._tile_positions(41, 16, 8) == [0, 8, 16, 24, 25]


def test_config_roundtrip():
    adapter = MedCAGDAdapter()
    args = SimpleNamespace(
        input_size=256,
        medcagd_encoder='pvt_v2_b2',
        medcagd_eval_stride=128,
        medcagd_eval_batch=8,
        medcagd_threshold=0.5,
        medcagd_edge_weight=1.0,
    )
    config = adapter.save_config(args)
    restored = MedCAGDAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert restored.input_size == 256
    assert target.medcagd_encoder == 'pvt_v2_b2'
    assert target.medcagd_eval_stride == 128
    assert target.medcagd_threshold == 0.5


@pytest.mark.slow
def test_model_forward():
    pytest.importorskip('timm')
    from baseline_zoo.models.medcagd import build_medcagd

    model = build_medcagd(input_size=64, num_classes=1, pretrained=False)
    output = model(torch.zeros(1, 3, 64, 64))
    main, auxiliaries, edges = output
    assert main.shape == (1, 1, 64, 64)
    assert len(auxiliaries) == 3 and len(edges) == 3
