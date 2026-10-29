"""Fast FSSAM adapter tests (no full SAM 2 run by default)."""

import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.fssam import FSSAMAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.models.fssam import Weighted_GAP, weighted_dice_loss
from baseline_zoo.models.fssam.sam2.modeling.sam.transformer import (
    RoPEAttention,
)


def _sample(tmp_path, name='TEST_0'):
    image = tmp_path / f'{name}_image.png'
    mask = tmp_path / f'{name}_mask.png'
    Image.new('RGB', (48, 40), (90, 110, 130)).save(image)
    label = Image.new('L', (48, 40), 0)
    for x in range(10, 36):
        label.putpixel((x, 20), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem=name)


def test_registry_and_metadata():
    assert 'fssam' in available_adapters()
    adapter = load_adapter('fssam')
    assert isinstance(adapter, FSSAMAdapter)
    assert adapter.id == 'fssam'
    assert adapter.input_size == 512
    assert adapter.support_based
    assert any('support-based' in item for item in adapter.deviations)


def test_weighted_gap_matches_mask_average():
    feats = torch.tensor([[[[1.0, 2.0], [3.0, 5.0]]]])
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    out = Weighted_GAP(feats, mask)
    assert out.shape == (1, 1, 1, 1)
    assert torch.isclose(out.flatten()[0], torch.tensor(1.5), atol=1e-3)


def test_weighted_dice_loss_is_finite_and_differentiable():
    logits = torch.zeros(2, 64, 64, requires_grad=True)
    target = torch.zeros(2, 64, 64)
    target[:, 20:30, 20:40] = 1
    loss = weighted_dice_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_augmentation_shapes_and_mask_values(tmp_path):
    adapter = FSSAMAdapter()
    adapter.input_size = 512
    adapter.args = SimpleNamespace(fssam_shots=3)
    image, mask = adapter._augment(
        *adapter._load_pair(_sample(tmp_path)), random.Random(0))
    assert image.shape == (3, 512, 512)
    assert mask.shape == (1, 512, 512)
    assert image.min() >= 0 and image.max() <= 255
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_train_supports_exclude_query(tmp_path):
    items = [_sample(tmp_path, f'TEST_{i}') for i in range(3)]
    adapter = FSSAMAdapter()
    adapter._support_groups = {'TEST': items}
    supports = adapter._train_supports(items[0], random.Random(0), 3)
    assert len(supports) == 2
    assert all(item.stem != items[0].stem for item in supports)


def test_scma_calibration_supports_multiple_shots():
    attn = RoPEAttention(
        embedding_dim=8, num_heads=2, downsample_rate=1, dropout=0.0,
        rope_k_repeat=True, feat_sizes=(4, 4), kv_in_dim=8)
    batch, shot, pixels, ptrs = 2, 2, 16, 2
    q = torch.randn(batch, pixels, 8, requires_grad=True)
    k = torch.randn(batch, shot * pixels + pixels + ptrs, 8)
    v = torch.randn(batch, shot * pixels + pixels + ptrs, 8)
    mask = torch.zeros(batch * shot, 1, 4, 4)
    mask[:, :, :2, :2] = 1
    out, weight = attn(
        q, k, v, num_k_exclude_rope=ptrs, memory_mask=mask,
        calibrate=True)
    assert out.shape == (batch, pixels, 8)
    assert weight is None  # the released SCMA branch returns no weights
    out.sum().backward()
    assert q.grad is not None


def test_config_roundtrip():
    adapter = FSSAMAdapter()
    args = SimpleNamespace(
        input_size=512,
        fssam_sam2_checkpoint='checkpoints/sam2_hiera_large.pt',
        fssam_sam2_config='sam2_hiera_l.yaml',
        fssam_dino_weights='checkpoints/dinov2_vitl14_pretrain.pth',
        fssam_dino_size='vitl',
        fssam_num_refine=3,
        fssam_ver_refine='v1',
        fssam_shots=3,
        fssam_support_dataset='auto',
        data_root='data/sacm_3shot_seed0',
    )
    config = adapter.save_config(args)
    restored = FSSAMAdapter()
    target = SimpleNamespace()
    restored.restore_config(target, config)
    assert target.fssam_sam2_config == 'sam2_hiera_l.yaml'
    assert target.fssam_num_refine == 3
    assert target.fssam_shots == 3


@pytest.mark.slow
def test_real_fssam_forward():
    from baseline_zoo.models.fssam import build_fssam

    sam2 = Path('checkpoints/sam2_hiera_small.pt')
    dino = Path('checkpoints/dinov2_vitb14_pretrain.pth')
    if not sam2.is_file() or not dino.is_file():
        pytest.skip('missing SAM 2 / DINOv2 checkpoints')
    model = build_fssam(
        sam2, sam2_config='sam2_hiera_s.yaml', dino_weights=dino,
        dino_size='vitb')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    query = torch.rand(1, 3, 512, 512, device=device) * 255
    supports = torch.rand(1, 1, 3, 512, 512, device=device) * 255
    masks = torch.zeros(1, 1, 512, 512, device=device)
    masks[:, :, 200:260, 100:400] = 1
    with torch.no_grad():
        logits = model(query, supports, masks)
    assert logits.shape == (1, 512, 512)
    assert torch.isfinite(logits).all()
