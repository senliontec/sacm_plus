"""Tests for training provenance, resumable artifacts, and audit indexes."""

import csv
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def artifacts():
    return _load('run_artifacts_under_test', 'src/core/run_artifacts.py')


def _image(path: Path, value: int = 1):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.full((8, 8), value, dtype=np.uint8) * 255).save(path)


def _data_root(tmp_path: Path) -> Path:
    root = tmp_path / 'data'
    for split in ('train', 'val'):
        _image(root / split / 'images' / 'DRIVE_a.png')
        _image(root / split / 'masks' / 'DRIVE_a.png')
    test_mask = root / 'test' / 'DRIVE' / 'masks' / 'DRIVE_1.png'
    _image(test_mask)
    with (root / 'split_manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['dataset', 'split', 'output', 'group'])
        writer.writeheader()
        writer.writerow({'dataset': 'DRIVE', 'split': 'train',
                         'output': 'DRIVE_a.png', 'group': 'DRIVE'})
    with (root / 'protocol_manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['dataset', 'split', 'mask_path', 'role'])
        writer.writeheader()
        writer.writerow({
            'dataset': 'DRIVE', 'split': 'seen_test',
            'mask_path': str(test_mask), 'role': 'seen'})
    return root


def test_data_manifest_and_stable_fingerprint(artifacts, tmp_path):
    root = _data_root(tmp_path)
    first = artifacts.build_data_manifest(root, hash_files=True)
    second = artifacts.build_data_manifest(root, hash_files=True)

    assert len(first['splits']['train']['images']) == 1
    assert len(first['splits']['val']['masks']) == 1
    assert first['test_datasets']['DRIVE']['n_masks'] == 1
    assert first['test_datasets']['DRIVE']['masks'][0]['sha256']
    assert artifacts.data_manifest_fingerprint(first) == (
        artifacts.data_manifest_fingerprint(second))


def test_prepare_training_artifacts_writes_provenance(artifacts, tmp_path):
    root = _data_root(tmp_path)
    save_dir = tmp_path / 'run'
    args = SimpleNamespace(
        data_root=root,
        protocol=None,
        checkpoint=None,
        epochs=3,
        preset='full',
        seed=0,
        api_key='should-not-appear',
    )
    info = artifacts.prepare_training_artifacts(
        args, save_dir, hash_files=False)
    assert Path(info['train_config']).is_file()
    assert Path(info['data_manifest']).is_file()
    assert (save_dir / 'provenance' / 'environment.json').is_file()
    assert (save_dir / 'provenance' / 'git.json').is_file()
    assert (save_dir / 'provenance' / 'protocol_manifest.csv').is_file()
    payload = json.loads(Path(info['train_config']).read_text())
    assert payload['args']['api_key'] == '***'
    assert payload['args']['preset'] == 'full'
    assert payload['data_manifest']['sha256'] == (
        info['data_manifest_sha256'])


def test_prediction_index_survives_relative_paths(artifacts, tmp_path):
    out_dir = tmp_path / 'eval' / 'DRIVE'
    out_dir.mkdir(parents=True)
    _image(out_dir / 'foo_result.png')
    gt = tmp_path / 'gt.png'
    _image(gt)
    rows = [{
        'dataset': 'DRIVE',
        'image': 'foo.png',
        'pred_path': 'foo_result.png',
        'gt_path': str(gt),
        'image_path': str(tmp_path / 'foo.png'),
    }]
    manifest = artifacts.write_prediction_index(
        out_dir, 'DRIVE', rows, manifest_extra={'metric_keys': ['dice']})
    assert Path(manifest['prediction_index']).is_file()
    with open(out_dir / 'prediction_index.csv', newline='') as handle:
        read_rows = list(csv.DictReader(handle))
    assert read_rows[0]['pred_path'] == 'foo_result.png'
    assert read_rows[0]['gt_path'] == str(gt)


def test_rng_state_round_trip(artifacts):
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    snap = artifacts.capture_rng_state()
    py_next = random.random()
    np_next = float(np.random.random())
    torch_next = float(torch.rand(()))
    artifacts.restore_rng_state(snap)
    assert random.random() == pytest.approx(py_next)
    assert float(np.random.random()) == pytest.approx(np_next)
    assert float(torch.rand(())) == pytest.approx(torch_next)


def test_eval_root_stays_in_checkpoint_variant_run(tmp_path):
    script = _load(
        'run_baseline_test_under_test',
        'scripts/pipeline/run_baseline_test.py')
    canonical = tmp_path / 'sacm_stc' / 'all_dense' / 'seed_0'
    # No --paper-root: canonical eval directory.
    assert script.resolve_eval_root(None, canonical, canonical) == (
        canonical / 'eval')
    # Config template passes the checkpoint's run root.
    assert script.resolve_eval_root(canonical, canonical, canonical) == (
        canonical / 'eval')
    # Explicit external output directory is preserved.
    external = tmp_path / 'custom_eval'
    assert script.resolve_eval_root(external, canonical, canonical) == external
    # A checkpoint copied outside results/runs still resolves beside itself.
    copied = tmp_path / 'copy' / 'all_dense' / 'seed_0'
    assert script.resolve_eval_root(copied, canonical, copied) == (
        copied / 'eval')


def test_checkpoint_fingerprint_resolves_nnunet_model_pointer(tmp_path):
    script = _load(
        'run_baseline_test_under_test',
        'scripts/pipeline/run_baseline_test.py')
    model_dir = tmp_path / 'model'
    fold = model_dir / 'fold_0'
    fold.mkdir(parents=True)
    checkpoint = fold / 'checkpoint_benchmark_best.pth'
    checkpoint.write_bytes(b'nnunet-benchmark-weights')
    pointer = tmp_path / 'nnunet_model_dir.txt'
    pointer.write_text(str(model_dir) + '\n')

    expected = script.file_sha256(checkpoint)
    assert script.checkpoint_fingerprint(pointer) == expected
    assert script.checkpoint_fingerprint(checkpoint) == expected
    assert script.checkpoint_fingerprint(tmp_path / 'missing.txt') is None


def test_finalizer_promotes_extra_artifacts(tmp_path):
    finalizer = _load('finalize_run_under_test',
                      'scripts/pipeline/finalize_run.py')
    source = tmp_path / 'raw' / 'full'
    run_root = tmp_path / 'run'
    source.mkdir(parents=True)
    run_root.mkdir()
    best = source / 'best_model.pth'
    torch.save({'model_state_dict': {}}, best)
    torch.save({'model_state_dict': {}}, source / 'last_model.pth')
    (source / 'train_config.json').write_text('{}\n')
    (source / 'wandb_run.json').write_text('{}\n')
    (source / 'provenance').mkdir()
    (source / 'provenance' / 'data_manifest.json').write_text('{}\n')
    (source / 'checkpoints').mkdir()
    torch.save({'epoch': 1}, source / 'checkpoints' / 'epoch_0001.pth')

    info = finalizer._promote_extra_artifacts(
        source, 'full', finalizer.CheckpointInfo(best, 'file'), run_root)
    assert not (run_root / 'last_model.pth').exists()
    assert not (run_root / 'checkpoints').exists()
    assert (run_root / 'train_config.json').is_file()
    assert (run_root / 'provenance' / 'data_manifest.json').is_file()
    assert (run_root / 'wandb_run.json').is_file()
    assert info['last_checkpoint'] is None
    assert info['checkpoints'] is None


def test_successful_training_cleanup_removes_temporary_checkpoints(tmp_path):
    from baseline_zoo.runtime import (
        _remove_temporary_checkpoints,
        run_prediction,
    )

    workdir = tmp_path / 'run'
    workdir.mkdir()
    last = workdir / 'last_model.pth'
    last.write_bytes(b'temporary')
    periodic = workdir / 'checkpoints'
    periodic.mkdir()
    (periodic / 'epoch_0001.pth').write_bytes(b'temporary')

    _remove_temporary_checkpoints(workdir, last)
    assert not last.exists()
    assert not periodic.exists()

    with pytest.raises(SystemExit):
        run_prediction(object(), SimpleNamespace(), last)
