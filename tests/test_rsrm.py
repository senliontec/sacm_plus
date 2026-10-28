"""Fast RSRM adapter tests (no real DINOv3 by default)."""

import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.adapters.rsrm import RSRMAdapter
from baseline_zoo.data import Sample
from baseline_zoo.models.rsrm.model import RSRM


def _sample(tmp_path, name='TEST_0'):
    image = tmp_path / f'{name}_image.png'
    mask = tmp_path / f'{name}_mask.png'
    Image.new('RGB', (64, 48), (90, 120, 150)).save(image)
    label = Image.new('L', (64, 48), 0)
    for x in range(8, 56):
        label.putpixel((x, 24), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem=name)


def test_registry_and_metadata():
    assert 'rsrm' in available_adapters()
    adapter = load_adapter('rsrm')
    assert isinstance(adapter, RSRMAdapter)
    assert adapter.id == 'rsrm'
    assert adapter.input_size == 480
    assert adapter.training_free
    assert adapter.support_based
    assert any('fast_pytorch_kmeans' in item for item in adapter.deviations)


def test_robust_threshold_and_prior_mask():
    prob = torch.zeros(1, 2, 8, 8)
    prob[:, 1, :4] = 0.9
    prob[:, 0, 4:] = 0.9
    target = torch.zeros(1, 8, 8)
    target[:, :4] = 1
    threshold = RSRM.find_robust_threshold_iou(prob[:, 1], target)
    assert 0.0 <= threshold <= 1.0
    prior = RSRM.generate_prior_mask(
        prob.softmax(1), fg_threshold=threshold, bg_threshold=threshold)
    assert prior.shape == (1, 2, 8, 8)
    assert set(prior.unique().tolist()) <= {0.0, 1.0}


def test_support_enhance_shape_and_grad_free():
    query = torch.randn(1, 16, 4, 4, requires_grad=True)
    support = torch.randn(1, 16, 4, 4)
    prior = torch.zeros(1, 2, 4, 4)
    prior[:, 0, :2] = 1
    prior[:, 1, 2:] = 1
    support_mask = torch.zeros(1, 4, 4, dtype=torch.long)
    support_mask[:, 2:] = 1
    out = RSRM.support_enhance(query, prior, support, support_mask)
    assert out.shape == (1, 16, 4, 4)
    out.sum().backward()
    assert query.grad is not None


def test_region_similarity_uses_kmeans():
    model = RSRM(backbone=torch.nn.Identity(), layers=1, seed=0)
    feature_s = torch.randn(30, 8)
    feature_q = torch.randn(1, 8, 4, 4)
    out = model.region_similarity(feature_s, feature_q, 5)
    assert out.shape == (1, 4, 4)
    assert torch.isfinite(out).all()


def test_timm_f_attn_matches_meta_xc():
    timm = pytest.importorskip('timm')
    from baseline_zoo.models.rsrm.model import _TimmDINOv3

    base = timm.create_model(
        'vit_small_patch16_dinov3', pretrained=False, num_classes=0).eval()
    wrapper = _TimmDINOv3(base, depth=len(base.blocks))
    x = torch.randn(1, 3, 64, 64)
    indices = list(range(len(base.blocks)))
    captured = {}

    with torch.no_grad():
        sums, attns = wrapper.get_intermediate_layers(
            x, n=indices, reshape=True)
        handles = [
            base.blocks[-1].attn.register_forward_hook(
                lambda _m, _i, out: captured.__setitem__('attn', out)),
            base.blocks[-1].register_forward_hook(
                lambda _m, _i, out: captured.__setitem__('block', out)),
        ]
        try:
            base(x)
        finally:
            for handle in handles:
                handle.remove()

        prefix = base.num_prefix_tokens
        value = captured['attn']
        gamma = getattr(base.blocks[-1], 'gamma_1', None)
        if gamma is not None:
            value = value * gamma
        reference_attn = base.norm(value)[:, prefix:]
        reference_attn = reference_attn.reshape(
            1, 4, 4, -1).permute(0, 3, 1, 2).contiguous()
        reference_sum = base.norm(captured['block'])[:, prefix:]
        reference_sum = reference_sum.reshape(
            1, 4, 4, -1).permute(0, 3, 1, 2).contiguous()

    assert torch.allclose(attns[-1], reference_attn, atol=1e-6)
    assert torch.allclose(sums[-1], reference_sum, atol=1e-6)


def test_pair_is_imagenet_normalized(tmp_path):
    adapter = RSRMAdapter()
    adapter.input_size = 480
    image, mask = adapter._pair(_sample(tmp_path))
    assert image.shape == (3, 480, 480)
    assert mask.shape == (480, 480)
    assert image.mean().abs() < 1.0
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_config_roundtrip():
    adapter = RSRMAdapter()
    args = SimpleNamespace(
        input_size=480,
        rsrm_dinov3_weights='checkpoints/dinov3/pytorch_model.bin',
        support_shots=3,
        support_dataset='auto',
        data_root='data/sacm_3shot_seed0',
    )
    config = adapter.save_config(args)
    restored = RSRMAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.input_size == 480
    assert target.support_shots == 3
    assert str(target.rsrm_dinov3_weights).endswith('pytorch_model.bin')


@pytest.mark.slow
def test_real_rsrm_forward():
    from baseline_zoo.models.rsrm import build_rsrm

    weights = Path('checkpoints/dinov3/pytorch_model.bin')
    if not weights.is_file():
        pytest.skip('missing DINOv3 checkpoint')
    model = build_rsrm(weights)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    supports = [torch.rand(1, 3, 480, 480, device=device) * 2 - 1
                for _ in range(2)]
    masks = []
    for _ in itertools.repeat(None, 2):
        mask = torch.zeros(1, 480, 480, device=device)
        mask[:, 100:200, 50:400] = 1
        masks.append(mask)
    query = torch.rand(1, 3, 480, 480, device=device) * 2 - 1
    with torch.no_grad():
        out = model(supports, masks, query)[0]
    assert out.shape == (1, 2, 480, 480)
    assert torch.isfinite(out).all()
