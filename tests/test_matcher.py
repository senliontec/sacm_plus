"""Fast Matcher adapter tests (no DINOv2/SAM build)."""

import inspect
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from baseline_zoo.adapters.matcher import MatcherAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.data import Sample
from baseline_zoo.runtime import run_training_free


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
    assert 'matcher' in available_adapters()
    adapter = load_adapter('matcher')
    assert isinstance(adapter, MatcherAdapter)
    assert adapter.id == 'matcher'
    assert adapter.training_free is True
    assert adapter.input_size == 518


def test_support_and_target_shapes(tmp_path):
    adapter = MatcherAdapter()
    adapter.args = SimpleNamespace(
        support_shots=3, support_dataset='auto')
    adapter._support_root = Path('data/sacm_3shot_seed0')
    images, masks = adapter._support_tensors('DRIVE', torch.device('cpu'))
    assert images.shape == (1, 3, 3, 518, 518)
    assert masks.shape == (1, 3, 518, 518)

    prepared = adapter.prepare_eval_sample(
        _sample(tmp_path), torch.device('cpu'))
    assert prepared['image'].shape == (1, 3, 518, 518)
    assert prepared['original_size'] == (24, 32)


def test_training_free_checkpoint(tmp_path):
    adapter = MatcherAdapter()
    root = Path(tempfile.mkdtemp(prefix='matcher_tf_'))
    args = SimpleNamespace(
        data_root=Path('data/sacm_3shot_seed0'), workdir=root,
        seed=0, device='cpu', batch_size=1, limit_train=0, limit_val=0,
        selection='dice', resume=False,
        sam_weights=Path('checkpoints/sam_vit_h_4b8939.pth'),
        dinov2_weights=Path('checkpoints/dinov2_vitl14_pretrain.pth'),
        matcher_random_init=True, support_shots=3,
        support_dataset='auto')
    ckpt = run_training_free(adapter, args)
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert payload['adapter'] == 'matcher'
    assert payload['model_state_dict'] == {}
    assert payload['config']['training_free'] is True
    assert payload['input_size'] == 518


def test_config_roundtrip(tmp_path):
    adapter = MatcherAdapter()
    args = SimpleNamespace(
        sam_weights=tmp_path / 'missing_sam.pth',
        dinov2_weights=tmp_path / 'missing_dino.pth',
        matcher_random_init=True, support_shots=3,
        support_dataset='auto', data_root=Path('data/sacm_3shot_seed0'))
    config = adapter.save_config(args)
    assert config['support_shots'] == 3
    assert config['matcher_random_init'] is True
    restore = SimpleNamespace()
    adapter.restore_config(restore, config)
    assert restore.support_shots == 3
    assert restore.matcher_random_init is True
    assert restore.input_size == 518


def test_official_fss_score_defaults():
    # Matcher.py imports POT at module level.  POT is an optional extra, so
    # keep this test importable on the documented `make install` / dev setup.
    pytest.importorskip('ot')
    from baseline_zoo.models.matcher.Matcher import Matcher
    from baseline_zoo.models.matcher.model import build_matcher

    matcher = inspect.signature(Matcher.__init__).parameters
    assert matcher['alpha'].default == 0.8
    assert matcher['beta'].default == 0.2
    assert matcher['exp'].default == 1.0
    builder = inspect.signature(build_matcher)
    # build_matcher hardcodes the official FSS score in the Matcher call.
    source = inspect.getsource(build_matcher)
    assert 'alpha=0.8' in source
    assert 'beta=0.2' in source
    assert 'exp=1.0' in source


def test_prediction_seeds_rng_once():
    adapter = MatcherAdapter()
    adapter.args = SimpleNamespace(seed=7)
    adapter._current_dataset = 'TEST'

    class FakeMatcher:
        def set_target(self, image):
            pass

        def predict(self):
            return torch.ones(1, 8, 8)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.matcher = FakeMatcher()

    torch.manual_seed(123)
    np.random.seed(123)
    result = adapter.predict_sample(
        FakeModel(), {'image': torch.zeros(1, 3, 8, 8),
                      'original_size': (8, 8)},
        SimpleNamespace(dataset='TEST'))
    assert result.shape == (8, 8) and result.all()
    assert adapter._prediction_seeded is True
    assert torch.initial_seed() == 7
    # Do not reseed here: these draws prove the adapter seeded the global
    # Python/NumPy RNGs, not that the test itself did.
    assert random.random() == pytest.approx(0.32383276483316237)
    assert np.random.randint(0, 2**31 - 1) == 327741615

    # A second call must not silently reseed (the once-only latch).  The
    # caller's 999 state must remain in effect.
    random.seed(999)
    np.random.seed(999)
    adapter.predict_sample(
        FakeModel(), {'image': torch.zeros(1, 3, 8, 8),
                      'original_size': (8, 8)},
        SimpleNamespace(dataset='TEST'))
    assert torch.initial_seed() == 7
    assert random.random() == pytest.approx(0.7813468849570298)
    assert np.random.randint(0, 2**31 - 1) == 1303213504
