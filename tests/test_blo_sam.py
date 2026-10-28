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
    clean_a = adapter._pair(_sample(tmp_path), random.Random(0),
                            augment=False)
    clean_b = adapter._pair(_sample(tmp_path), random.Random(99),
                            augment=False)
    assert torch.equal(clean_a[0], clean_b[0])
    assert torch.equal(clean_a[1], clean_b[1])


def test_bilevel_split_is_fixed_disjoint_and_stratified(tmp_path):
    adapter = BLOSAMAdapter()
    samples = [
        Sample(image_path=tmp_path / f'{domain}_{i}.png',
               mask_path=tmp_path / f'{domain}_{i}_m.png',
               dataset=domain, stem=f'{domain}_{i}')
        for domain in ('DOMAIN_A', 'DOMAIN_B')
        for i in range(3)
    ]
    d1, d2 = adapter._bilevel_split(samples, seed=7)
    assert len(d1) == 3 and len(d2) == 3
    assert not ({s.stem for s in d1} & {s.stem for s in d2})
    assert {s.stem for s in d1} | {s.stem for s in d2} == {
        s.stem for s in samples}
    # Every pooled domain must appear in both bilevel levels.
    assert {s.dataset for s in d1} == {'DOMAIN_A', 'DOMAIN_B'}
    assert {s.dataset for s in d2} == {'DOMAIN_A', 'DOMAIN_B'}
    d1b, d2b = adapter._bilevel_split(list(reversed(samples)), seed=7)
    assert [s.stem for s in d1b] == [s.stem for s in d1]
    assert [s.stem for s in d2b] == [s.stem for s in d2]


def test_predict_upsamples_foreground_logits(tmp_path):
    adapter = BLOSAMAdapter()
    adapter.input_size = 64
    adapter._multimask_output = False

    class _FakeSAM(torch.nn.Module):
        def forward(self, image, multimask_output, image_size):
            batch, _, height, width = image.shape
            logits = torch.full((batch, 2, height, width), -10.0)
            logits[:, 1, height // 2:height // 2 + 2, :] = 10.0
            return {'masks': logits, 'low_res_logits': logits}

    image = torch.zeros(1, 3, 64, 64)
    sample = _sample(tmp_path)
    pred = adapter.predict_sample(
        _FakeSAM(), (image, (32, 48)), sample)
    assert pred.shape == (48, 32)  # (H, W) from prepared size
    assert pred.dtype == bool
    assert pred.any()


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
