"""Fast SCRWKV tests (dummy WKV; real CUDA kernel is optional)."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import baseline_zoo.models.scrwkv.SCIU as SCIU
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.adapters.scrwkv import SCRWKVAdapter
from baseline_zoo.data import Sample
from baseline_zoo.models.scrwkv import build_scrwkv


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
    assert 'scrwkv' in available_adapters()
    adapter = load_adapter('scrwkv')
    assert isinstance(adapter, SCRWKVAdapter)
    assert adapter.id == 'scrwkv'
    assert adapter.input_size == 512
    assert adapter.official_lr == pytest.approx(5e-4)
    assert any('bi_wkv' in item for item in adapter.deviations)


def test_model_build_and_parameter_count():
    model = build_scrwkv()
    count = sum(p.numel() for p in model.parameters())
    assert 1_500_000 < count < 2_500_000
    assert all(p.requires_grad for p in model.parameters())


def test_adapter_pair_and_loss(tmp_path):
    adapter = SCRWKVAdapter()
    adapter.input_size = 512
    image, mask = adapter._pair(_sample(tmp_path))
    assert image.shape == (3, 512, 512)
    assert mask.shape == (1, 512, 512)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert mask.sum() > 0
    assert image.min() >= -1.0 and image.max() <= 1.0
    logits = torch.zeros(1, 1, 64, 64, requires_grad=True)
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, 20:30, 20:40] = 1
    loss = adapter._bce_dice(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_forward_with_dummy_wkv(monkeypatch):
    monkeypatch.setattr(SCIU, 'RUN_CUDA', lambda w, u, k, v: k * v)
    model = build_scrwkv().eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 512, 512))
    assert out.shape == (1, 1, 512, 512)
    assert torch.isfinite(out).all()


def test_config_roundtrip():
    adapter = SCRWKVAdapter()
    args = SimpleNamespace(input_size=512,
                           data_root='data/sacm_3shot_seed0')
    config = adapter.save_config(args)
    restored = SCRWKVAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 512


@pytest.mark.slow
def test_real_cuda_wkv_forward_if_available():
    if SCIU.wkv_cuda is None:
        pytest.skip('SCRWKV CUDA extension not built on this host')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    model = build_scrwkv().cuda().eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 512, 512, device='cuda'))
    assert out.shape == (1, 1, 512, 512)
