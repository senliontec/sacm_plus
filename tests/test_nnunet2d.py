"""Fast nnU-Net 2D adapter tests (no nnU-Net training)."""

import json

import pytest

from baseline_zoo.adapters.nnunet2d import NNUNet2DAdapter
from baseline_zoo.adapters.registry import available_adapters, load_adapter
from baseline_zoo.registry import load_models


def test_registry_and_metadata():
    assert 'nnunet2d' in available_adapters()
    adapter = load_adapter('nnunet2d')
    assert isinstance(adapter, NNUNet2DAdapter)
    assert adapter.id == 'nnunet2d'
    assert adapter.custom_pipeline is True
    assert adapter.official_lr == pytest.approx(1e-2)
    assert any('18-image' in item for item in adapter.deviations)
    model = load_models()['nnunet2d']
    assert model.checkpoint_template.endswith('nnunet_model_dir.txt')
    assert '--data-root' in model.eval_entrypoint


def test_case_id_is_sanitized_and_unique():
    used = set()
    first = NNUNet2DAdapter._case_id('DRIVE_31 training', used)
    used.add(first)
    second = NNUNet2DAdapter._case_id('DRIVE_31 training', used)
    assert first == 'DRIVE_31_training'
    assert second == 'DRIVE_31_training_1'


def test_write_splits(tmp_path):
    NNUNet2DAdapter._write_splits(tmp_path, ['a', 'b'], ['c'])
    data = json.loads((tmp_path / 'splits_final.json').read_text())
    assert data == [{'train': ['a', 'b'], 'val': ['c']}]


def test_find_checkpoint_prefers_best(tmp_path):
    fold = tmp_path / 'fold_0'
    fold.mkdir()
    (fold / 'checkpoint_final.pth').write_bytes(b'x')
    (fold / 'checkpoint_best.pth').write_bytes(b'x')
    assert NNUNet2DAdapter._find_checkpoint(
        tmp_path, 'best') == fold / 'checkpoint_best.pth'
    (fold / 'checkpoint_best.pth').unlink()
    assert NNUNet2DAdapter._find_checkpoint(tmp_path, 'best') is None
    (fold / 'checkpoint_best.safetensors').write_bytes(b'x')
    assert NNUNet2DAdapter._find_checkpoint(
        tmp_path, 'best') == fold / 'checkpoint_best.safetensors'


def test_resume_is_rejected_before_staging():
    adapter = NNUNet2DAdapter()
    args = type('Args', (), {})()
    args.resume = True
    with pytest.raises(SystemExit, match='resume'):
        adapter.run_training(args)
