"""Fast BLO-SAM adapter tests (real model is slow-gated)."""

import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.blo_sam import BLOSAMAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample

REPO_ROOT = Path(__file__).resolve().parents[1]
SAM_CKPT = REPO_ROOT / 'checkpoints' / 'sam_vit_b_01ec64.pth'


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
    assert 'blo_sam' in available_adapters()
    adapter = load_adapter('blo_sam')
    assert isinstance(adapter, BLOSAMAdapter)
    assert adapter.id == 'blo_sam'
    assert adapter.input_size == 256
    assert adapter.train_batch_size == 1
    assert any('bilevel' in item.lower() for item in adapter.deviations)


def test_pair_shapes(tmp_path):
    adapter = BLOSAMAdapter()
    adapter.input_size = 64
    image, mask, low = adapter._pair(_sample(tmp_path), random.Random(0))
    assert image.shape == (3, 64, 64)
    assert mask.shape == (64, 64)
    assert low.shape == (16, 16)
    assert set(mask.unique().tolist()) <= {0, 1}
    assert set(low.unique().tolist()) <= {0, 1}


def test_config_roundtrip():
    adapter = BLOSAMAdapter()
    args = SimpleNamespace(
        input_size=256,
        blo_sam_rank=4,
        blo_sam_dice_weight=0.8,
        blo_sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',
    )
    config = adapter.save_config(args)
    restored = BLOSAMAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert restored.input_size == 256
    assert target.blo_sam_rank == 4
    assert target.blo_sam_dice_weight == 0.8


@pytest.mark.slow
def test_model_forward_and_loss(tmp_path):
    pytest.importorskip('torch')
    if not SAM_CKPT.is_file():
        pytest.skip(f'missing {SAM_CKPT}')
    adapter = BLOSAMAdapter()
    adapter.device = torch.device('cpu')
    args = SimpleNamespace(
        input_size=64,
        blo_sam_checkpoint=SAM_CKPT,
        blo_sam_rank=4,
        blo_sam_dice_weight=0.8,
        stage='train',
    )
    model = adapter.build_model(args)
    adapter.args = args
    batch = adapter.prepare_train_batch(
        [_sample(tmp_path)], random.Random(0))
    # The SAM ViT-B image encoder requires a 16-pixel multiple; 64 is valid.
    loss = adapter.train_step(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters()
               if p.requires_grad)
