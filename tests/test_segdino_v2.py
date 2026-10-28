"""Fast SegDINO v2 tests (real DINOv3 forward is slow/optional)."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.adapters.segdino_v2 import SegDINOV2Adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.segdino_v2 import build_segdino_v2


def _sample(tmp_path):
    image = tmp_path / 'TEST_0_image.png'
    mask = tmp_path / 'TEST_0_mask.png'
    Image.new('RGB', (64, 48), (120, 120, 120)).save(image)
    label = Image.new('L', (64, 48), 0)
    for x in range(8, 56):
        label.putpixel((x, 24), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem='TEST_0')


def test_registry_and_metadata():
    assert 'segdino_v2' in available_adapters()
    adapter = load_adapter('segdino_v2')
    assert isinstance(adapter, SegDINOV2Adapter)
    assert adapter.id == 'segdino_v2'
    assert adapter.input_size == 256
    assert adapter.official_lr == pytest.approx(1e-4)
    assert any('frozen encoder' in item for item in adapter.deviations)


def test_random_build_forward_and_freeze_policy():
    model = build_segdino_v2(random_init=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)
    adapter = SegDINOV2Adapter()
    adapter.args = SimpleNamespace(segdino_v2_train_backbone=False)
    adapter.configure_trainable(model)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert any(p.requires_grad for p in model.decoder.parameters())


def test_pair_shapes_and_normalization(tmp_path):
    adapter = SegDINOV2Adapter()
    adapter.input_size = 256
    adapter.args = SimpleNamespace(segdino_v2_no_augment=True)
    image, mask = adapter._resize_pair(_sample(tmp_path), None, train=True)
    assert image.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_loss_is_finite_and_differentiable():
    from baseline_zoo.models.segdino_v2 import build_segdino_v2
    model = build_segdino_v2(random_init=True)
    model.eval()
    logits = model(torch.randn(1, 3, 64, 64))
    logits = logits.requires_grad_()
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, 20:40, 20:50] = 1
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_config_roundtrip():
    adapter = SegDINOV2Adapter()
    args = SimpleNamespace(
        input_size=256,
        segdino_v2_dino_size='small',
        segdino_v2_dino_ckpt=None,
        segdino_v2_decoder_dim=256,
        segdino_v2_train_backbone=False,
        segdino_v2_no_augment=False,
        segdino_v2_random_init=False,
        epochs=800,
        data_root='data/sacm_3shot_seed0',
    )
    config = adapter.save_config(args)
    restored = SegDINOV2Adapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 256
    assert target.segdino_v2_decoder_dim == 256
    assert target.segdino_v2_no_augment is False


@pytest.mark.slow
def test_real_dinov3_forward_if_present():
    model = build_segdino_v2()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)
