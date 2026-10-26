"""Fast UCS adapter tests (no ViT-L build)."""

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_eval_preparation_seeds_initial_prompt(tmp_path):
    sample = _sample(tmp_path)

    def prepare(prior_seed):
        adapter = UCSAdapter()
        adapter.input_size = 1024
        adapter.args = SimpleNamespace(seed=5, stage='predict')
        np.random.seed(prior_seed)
        return adapter, adapter.prepare_eval_sample(
            sample, torch.device('cpu'))

    first_adapter, first = prepare(999)
    second_adapter, second = prepare(111)
    assert first_adapter._prediction_seeded is True
    assert second_adapter._prediction_seeded is True
    assert torch.equal(first['image'], second['image'])
    assert torch.equal(first['boxes'], second['boxes'])
    assert torch.equal(first['point_coords'], second['point_coords'])
    assert torch.equal(first['point_labels'], second['point_labels'])


def test_training_validation_does_not_reseed_rng(tmp_path):
    adapter = UCSAdapter()
    adapter.input_size = 1024
    adapter.args = SimpleNamespace(seed=5, stage='train')
    adapter.prepare_eval_sample(_sample(tmp_path), torch.device('cpu'))
    assert adapter._prediction_seeded is False


def test_prediction_seeds_rng_once():
    adapter = UCSAdapter()
    adapter.iter_point = 1
    adapter.point_num = 1
    adapter.device = torch.device('cpu')
    adapter.args = SimpleNamespace(seed=5, stage='predict')

    class FakeEncoder:
        def __call__(self, images):
            return torch.zeros(1, 1), []

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.image_encoder = FakeEncoder()

    def fake_decode(model, batch, image_embeddings, interm_embeddings):
        return torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), None

    adapter._decode = fake_decode
    prepared = {
        'image': torch.zeros(1, 3, 1024, 1024),
        'label': torch.zeros(1, 1, 1024, 1024),
        'boxes': None,
        'point_coords': torch.zeros(1, 1, 2),
        'point_labels': torch.ones(1, 1, dtype=torch.int),
        'mask_inputs': None,
        'original_size': (4, 4),
    }
    torch.manual_seed(123)
    np.random.seed(123)
    result = adapter.predict_sample(FakeModel(), prepared, _sample(Path('/tmp')))
    assert result.shape == (4, 4) and not result.any()
    assert adapter._prediction_seeded is True
    assert torch.initial_seed() == 5
    np.random.seed(5)
    assert np.random.randint(0, 2**31 - 1) == 953453411
