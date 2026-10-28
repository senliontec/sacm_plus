"""Fast DKA adapter tests (random init only)."""

import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.dka import DKAAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.dka import DKAModule, build_dka


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
    assert 'dka' in available_adapters()
    adapter = load_adapter('dka')
    assert isinstance(adapter, DKAAdapter)
    assert adapter.id == 'dka'
    assert adapter.input_size == 512
    assert any('Segmenter' in item for item in adapter.deviations)


def test_dka_starts_as_identity_and_has_gradients():
    module = DKAModule(16, middle_dim=4)
    x = torch.randn(1, 10, 16)
    out = module(x)
    assert torch.allclose(out, x)
    with torch.no_grad():
        module.adapter_down.weight.normal_()
        module.adapter_up.weight.normal_()
    out = module(x)
    assert not torch.allclose(out, x)
    out.sum().backward()
    assert module.adapter_up.weight.grad is not None
    assert module.adapter_down.weight.grad is not None
    assert module.conv_large.weight.grad is not None


def test_segmenter_forward_and_trainable_policy():
    model = build_dka(None)
    assert model.vit.pos_embed.shape == (1, 1025, 768)
    assert hasattr(model.vit.layers[0], 'adapter')
    model.train()
    assert model.vit.layers[-1].block.drop_path.training
    image = torch.zeros(1, 3, 512, 512)
    logits = model(image)
    assert logits.shape == (1, 2, 512, 512)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable
    assert not any(
        p.requires_grad for p in model.vit.patch_embed.parameters())
    assert not any(p.requires_grad for p in model.vit.ln1.parameters())
    assert any(p.requires_grad for p in model.decoder.parameters())


def test_pair_shapes_and_normalization(tmp_path):
    adapter = DKAAdapter()
    adapter.input_size = 512
    image, mask = adapter._pair(_sample(tmp_path), random.Random(0))
    assert image.shape == (3, 512, 512)
    assert mask.shape == (512, 512)
    assert mask.dtype == torch.long
    assert set(mask.unique().tolist()) <= {0, 1}
    # Grayscale 120 -> (120/255 - 0.5) / 0.5 = -0.0588235...
    assert image.mean() == pytest.approx(-0.0588235, abs=1e-5)


def test_config_roundtrip():
    adapter = DKAAdapter()
    args = SimpleNamespace(
        input_size=512,
        dka_vit_weights='checkpoints/segmenter_vit_b16_384.pth',
        dka_middle_dim=192,
        dka_adapter_lr=1e-3,
        dka_random_init=False,
        data_root='data/sacm_3shot_seed0',
    )
    config = adapter.save_config(args)
    restored = DKAAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 512
    assert target.dka_middle_dim == 192
    assert target.dka_adapter_lr == pytest.approx(1e-3)
    assert str(target.dka_vit_weights).endswith(
        'segmenter_vit_b16_384.pth')


@pytest.mark.slow
def test_real_segmenter_backbone_if_present():
    weights = Path('checkpoints/segmenter_vit_b16_384.pth')
    if not weights.is_file():
        pytest.skip('missing Segmenter-B checkpoint')
    model = build_dka(weights)
    assert model.vit.pos_embed.shape == (1, 1025, 768)
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 512, 512))
    assert logits.shape == (1, 2, 512, 512)
