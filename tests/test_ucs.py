"""Fast UCS adapter tests (no ViT-L build)."""

import random
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.adapters.ucs import UCSAdapter
from baseline_zoo.data import Sample

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sample(tmp_path):
    image = tmp_path / 'TEST_0_image.png'
    mask = tmp_path / 'TEST_0_mask.png'
    Image.new('RGB', (32, 24), (120, 120, 120)).save(image)
    label = Image.new('L', (32, 24), 0)
    for x in range(16):
        for y in range(24):
            label.putpixel((x, y), 255)
    label.save(mask)
    return Sample(image_path=image, mask_path=mask,
                  dataset='TEST', stem='TEST_0')


def test_registry_and_metadata():
    assert 'ucs' in available_adapters()
    adapter = load_adapter('ucs')
    assert adapter.id == 'ucs'
    assert adapter.input_size == 1024
    assert adapter.official_epochs == 20
    assert 'Oracle-prompted' in adapter.deviations[0]


def test_prompt_helpers():
    import numpy as np
    from baseline_zoo.models.ucs import utils

    mask = np.zeros((24, 32), dtype=np.float32)
    mask[:, :16] = 1
    boxes = utils.get_boxes_from_mask(mask, max_pixel=0)
    assert boxes.shape == (1, 4)
    coords, labels = utils.init_point_sampling(mask, 1)
    assert coords.shape == (1, 2)
    assert labels.shape == (1,)


def test_prepare_sample_shapes(tmp_path):
    adapter = UCSAdapter()
    batch = adapter._prepare_sample(
        _sample(tmp_path), True, random.Random(0))
    assert batch['image'].shape == (1, 3, 1024, 1024)
    assert batch['label'].shape == (1, 1, 1024, 1024)
    assert batch['original_size'] == (24, 32)
    assert ((batch['boxes'] is not None)
            or (batch['point_coords'] is not None))


def test_loss_and_config_roundtrip(tmp_path):
    from baseline_zoo.models.ucs.utils import FocalDiceloss_IoULoss

    criterion = FocalDiceloss_IoULoss()
    logits = torch.randn(1, 1, 64, 64, requires_grad=True)
    target = torch.zeros(1, 1, 64, 64)
    target[..., :32] = 1
    loss = criterion(logits, target, torch.zeros(1, 1))
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None

    adapter = UCSAdapter()
    adapter.input_size = 1024
    adapter.iter_point = 3
    adapter.point_num = 2
    adapter.mask_num = 1
    adapter.multimask = False
    adapter._sam_sha = 'abc'
    args = SimpleNamespace(sam_checkpoint=tmp_path / 'missing.pth')
    config = adapter.save_config(args)
    assert config['iter_point'] == 3
    assert config['sam_checkpoint_sha256'] == 'abc'
    restore = SimpleNamespace(input_size=0, sam_checkpoint=None)
    adapter.restore_config(restore, config)
    assert restore.input_size == 1024
    assert adapter.iter_point == 3
    assert adapter.point_num == 2


def test_pca_uses_device_not_cuda():
    source = (REPO_ROOT / 'src' / 'baseline_zoo' / 'models' / 'ucs'
              / 'segment_anything' / 'modeling' / 'mask_decoder.py').read_text()
    assert 'to(mask_tokens_out.device)' in source
    assert '.cuda().reshape(-1)' not in source
