"""Fast PerSAM adapter tests (no SAM ViT-H inference)."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baseline_zoo.adapters.persam import PerSAMAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.models.persam import (
    majority_vote,
    point_selection,
    prepare_reference_embedding,
    predict_with_embedding,
)
from baseline_zoo.runtime import run_training_free


def test_registry_and_metadata():
    assert 'persam' in available_adapters()
    adapter = load_adapter('persam')
    assert isinstance(adapter, PerSAMAdapter)
    assert adapter.id == 'persam'
    assert adapter.training_free is True
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


def test_reference_embedding_split_matches_official():
    class FakePredictor:
        def set_image(self, image, mask):
            self.features = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]],
                                          [[2.0, 4.0], [6.0, 8.0]],
                                          [[1.0, 1.0], [1.0, 1.0]],
                                          [[0.0, 1.0], [2.0, 3.0]]]])
            return torch.tensor([[[[1.0, 0.0], [0.0, 0.0]],
                                  [[1.0, 0.0], [0.0, 0.0]],
                                  [[1.0, 0.0], [0.0, 0.0]]]])

    similarity, raw = prepare_reference_embedding(
        FakePredictor(), np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2, 3), dtype=np.uint8))
    assert similarity.shape == (1, 1, 4)
    assert raw.shape == (1, 1, 4)
    assert similarity.norm().item() == pytest.approx(1.0)
    assert raw.norm().item() == pytest.approx(6.0 ** 0.5)
    assert torch.allclose(
        similarity, raw / raw.norm(dim=-1, keepdim=True))


def test_predict_passes_raw_embedding_to_target_prompting():
    class FakeModel:
        def postprocess_masks(self, sim, input_size, original_size):
            return sim

    class FakePredictor:
        def __init__(self):
            self.model = FakeModel()
            self.calls = []

        def predict(self, point_coords, point_labels, multimask_output,
                    **kwargs):
            self.calls.append(kwargs)
            masks = np.zeros((1, 8, 8), dtype=bool)
            masks[0, 2:5, 2:5] = True
            scores = np.array([0.9])
            logits = torch.zeros(1, 1, 8, 8)
            return masks, scores, logits, None

    predictor = FakePredictor()
    features = torch.randn(1, 4, 2, 2)
    similarity = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]])
    raw = torch.tensor([[[2.0, 2.0, 2.0, 2.0]]])
    result = predict_with_embedding(
        predictor, features, (8, 8), (8, 8), similarity, raw)
    assert result.shape == (8, 8) and result.any()
    assert predictor.calls[0]['target_embedding'] is raw
    assert predictor.calls[0]['attn_sim'] is not None
