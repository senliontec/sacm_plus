"""Fast PerSAM adapter tests (no SAM ViT-H inference)."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from baseline_zoo.adapters.persam import PerSAMAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.models.persam import majority_vote, point_selection
from baseline_zoo.runtime import run_training_free


def test_registry_and_metadata():
    assert 'persam' in available_adapters()
    adapter = load_adapter('persam')
    assert isinstance(adapter, PerSAMAdapter)
    assert adapter.id == 'persam'
    assert adapter.training_free is True
    assert adapter.official_epochs == 0
    assert 'one-shot' in adapter.deviations[0]


def test_point_selection():
    sim = torch.zeros(10, 8)
    sim[3, 5] = 1.0
    sim[7, 1] = -1.0
    top_xy, top_labels, last_xy, last_labels = point_selection(sim, topk=1)
    assert top_xy.shape == (1, 2)
    assert last_xy.shape == (1, 2)
    assert top_labels.tolist() == [1]
    assert last_labels.tolist() == [0]
    assert top_xy[0].tolist() == [5, 3]


def test_majority_vote():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    c = np.zeros((4, 4), dtype=bool)
    a[:2] = True
    b[:2] = True
    c[2:] = True
    voted = majority_vote([a, b, c])
    assert voted[:2].all()
    assert not voted[2:].any()


def test_training_free_checkpoint(tmp_path):
    adapter = PerSAMAdapter()
    root = Path(tempfile.mkdtemp(prefix='persam_tf_'))
    args = SimpleNamespace(
        data_root=Path('data/sacm_3shot_seed0'), workdir=root,
        seed=0, device='cpu', batch_size=1, limit_train=0, limit_val=0,
        selection='dice', resume=False,
        sam_checkpoint=Path('checkpoints/sam_vit_h_4b8939.pth'),
        sam_type='vit_h', support_shots=3, support_dataset='auto')
    ckpt = run_training_free(adapter, args)
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert payload['adapter'] == 'persam'
    assert payload['model_state_dict'] == {}
    assert payload['config']['training_free'] is True
    assert payload['input_size'] == 1024


def test_config_roundtrip(tmp_path):
    adapter = PerSAMAdapter()
    adapter._sam_sha = 'abc'
    args = SimpleNamespace(
        sam_checkpoint=tmp_path / 'missing.pth', sam_type='vit_h',
        support_shots=3, support_dataset='auto',
        data_root=Path('data/sacm_3shot_seed0'))
    config = adapter.save_config(args)
    assert config['support_shots'] == 3
    assert config['sam_checkpoint_sha256'] == 'abc'
    restore = SimpleNamespace()
    adapter.restore_config(restore, config)
    assert restore.support_shots == 3
    assert restore.sam_type == 'vit_h'
    assert restore.input_size == 1024
