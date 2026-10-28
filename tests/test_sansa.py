"""Fast SANSA adapter tests (no full SAM 2 run by default)."""

import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.adapters.sansa import SANSAAdapter
from baseline_zoo.data import Sample
from baseline_zoo.models.sansa import focal_dice_loss


def _sample(tmp_path, name='TEST_0'):
    image = tmp_path / f'{name}_image.png'
    mask = tmp_path / f'{name}_mask.png'
    Image.new('RGB', (64, 48), (120, 100, 80)).save(image)
    label = Image.new('L', (64, 48), 0)
    for x in range(8, 56):
        label.putpixel((x, 24), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem=name)


def test_registry_and_metadata():
    assert 'sansa' in available_adapters()
    adapter = load_adapter('sansa')
    assert isinstance(adapter, SANSAAdapter)
    assert adapter.id == 'sansa'
    assert adapter.input_size == 1024
    assert adapter.support_based
    assert any('AdaptFormer' in item for item in adapter.deviations)


def test_focal_dice_loss_is_finite_and_differentiable():
    logits = torch.zeros(2, 64, 64, requires_grad=True)
    masks = torch.zeros(1, 2, 64, 64)
    masks[:, 1, 20:30, 20:40] = 1
    loss = focal_dice_loss(logits, masks, num_frames=1)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_pair_is_imagenet_normalized(tmp_path):
    adapter = SANSAAdapter()
    adapter.input_size = 1024
    image, mask = adapter._pair(_sample(tmp_path))
    assert image.shape == (3, 1024, 1024)
    assert mask.shape == (1024, 1024)
    assert image.mean().abs() < 1.0
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_predict_sample_returns_gt_resolution_bool_mask():
    adapter = SANSAAdapter()
    adapter.args = SimpleNamespace(sansa_shots=3)
    adapter.device = torch.device('cpu')

    class _Fake(torch.nn.Module):
        def forward(self, samples, prompt):
            return torch.zeros(samples.shape[1], 1, 8, 8)

    prepared = {
        'samples': torch.zeros(1, 4, 3, 8, 8),
        'prompt': {'shots': 3, 'prompt_type': 'mask'},
        'size': (16, 16),
    }
    pred = adapter.predict_sample(_Fake(), prepared, None)
    assert pred.shape == (16, 16)
    assert pred.dtype == bool


def test_train_support_excludes_query(tmp_path):
    items = [_sample(tmp_path, f'TEST_{i}') for i in range(3)]
    adapter = SANSAAdapter()
    adapter._support_groups = {'TEST': items}
    support = adapter._train_support(items[0], random.Random(0))
    assert support.stem != items[0].stem


def test_prompt_dict_shapes(tmp_path):
    adapter = SANSAAdapter()
    masks = torch.zeros(1, 3, 1024, 1024)
    masks[:, 1, 100:200, 100:200] = 1
    prompt = adapter._prompt_dict(masks, 3)
    assert prompt['shots'] == 3
    assert prompt['prompt_type'] == 'mask'
    assert prompt[0][2]['prompt'].shape == (1, 1, 1024, 1024)


def test_config_roundtrip():
    adapter = SANSAAdapter()
    args = SimpleNamespace(
        input_size=1024,
        sansa_sam2_checkpoint='checkpoints/sam2_hiera_large.pt',
        sansa_sam2_config='sam2_hiera_l.yaml',
        sansa_adaptformer_stages=[2, 3],
        sansa_channel_factor=0.3,
        sansa_shots=3,
        sansa_support_dataset='auto',
        data_root='data/sacm_3shot_seed0',
    )
    config = adapter.save_config(args)
    restored = SANSAAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.sansa_sam2_config == 'sam2_hiera_l.yaml'
    assert target.sansa_adaptformer_stages == [2, 3]
    assert target.sansa_channel_factor == pytest.approx(0.3)


@pytest.mark.slow
def test_real_sansa_forward():
    from baseline_zoo.models.sansa import build_sansa

    checkpoint = Path('checkpoints/sam2_hiera_large.pt')
    if not checkpoint.is_file():
        pytest.skip('missing SAM 2 checkpoint')
    model = build_sansa(checkpoint)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert 9_000_000 < sum(p.numel() for p in trainable) < 11_000_000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    samples = torch.rand(1, 2, 3, 1024, 1024, device=device) * 255
    masks = torch.zeros(1, 2, 1024, 1024, device=device)
    masks[:, :, 400:520, 200:800] = 1
    prompt = {
        'shots': 1, 'prompt_type': 'mask',
        0: {0: {'prompt': masks[:, 0:1], 'prompt_type': 'mask'}},
    }
    with torch.no_grad():
        logits = model(samples, prompt)
    assert logits.shape == (2, 1024, 1024)
    assert torch.isfinite(logits).all()
