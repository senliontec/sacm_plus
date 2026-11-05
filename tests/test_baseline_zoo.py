"""Tests for the model registry, protocol expansion and dry-run planner."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.__main__ import (  # noqa: E402
    _eval_artifact,
    _run_job,
    _write_marker,
    main,
)
from baseline_zoo.protocols import load_protocol  # noqa: E402
from baseline_zoo.registry import load_models, resolve_models  # noqa: E402
from baseline_zoo.runtime import _select_cuda_device  # noqa: E402


def test_registry_contains_all_self_contained_models():
    models = load_models()
    expected = {
        'a2sam', 'aco_moe_lora', 'bcunet', 'blo_sam', 'c2p', 'convlora',
        'cs2net', 'csunet',
        'cwsam', 'dconnnet', 'dscnet', 'dka', 'flowsdf', 'fssam', 'jtfn',
        'knowsam', 'kprism', 'liot', 'maskvsc', 'matcher', 'medcagd',
        'nnunet2d',
        'nnwnet',
        'persam',
        'rsrm', 's2mnet', 'sacm_official', 'sacm_plus', 'skein',
        'sam2_unet',
        'sam_adapter', 'sam_med2d', 'sam_octa', 'sam_vit_l_unet', 'samed',
        'sansa',
        'scrwkv', 'segdino', 'segdino_v2', 'topolora', 'curvfield',
        'tpsegformer', 'ucs', 'unet'}
    assert set(models) == expected
    available = {model.id for model in models.values() if model.available}
    assert available == expected
    assert models['sam2_unet'].status == 'available'
    assert 'run_one' in models['sam2_unet'].train_entrypoint
    assert '--sam2-checkpoint' in models['sam2_unet'].train_entrypoint
    assert models['bcunet'].status == 'available'
    assert models['cwsam'].status == 'available'
    assert models['sam_med2d'].status == 'available'
    assert 'prompt' in models['sam_med2d'].note.lower()
    assert 'scripts/skein/run_skein_quick.sh' in models['skein'].train_entrypoint
    assert 'scripts/pipeline/run_baseline_test.py' in models['unet'].eval_entrypoint
    assert models['unet'].checkpoint_template == '{output_root}/best_model.pth'
    assert 'run_one' in models['unet'].train_entrypoint
    assert 'run_one' in models['knowsam'].train_entrypoint


def test_plan_expands_sam_octa_command(capsys):
    rc = main(['plan', '--model', 'sam_octa', '--seeds', '0'])
    output = capsys.readouterr().out
    assert rc == 0
    assert '--model sam_octa' in output
    assert 'run_one' in output
    assert 'results/runs/sam_octa/default/seed_0' in output


def test_report_dry_run_tolerates_missing_runs(tmp_path, capsys):
    rc = main(['report', '--model', 'unet', '--seeds', '0',
               '--result-root', str(tmp_path / 'empty'), '--dry-run'])
    assert rc == 0
    assert 'no metric rows' in capsys.readouterr().out


def test_eval_resume_requires_eval_artifact(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import baseline_zoo.__main__ as cli

    calls = []
    monkeypatch.setattr(
        cli, '_run',
        lambda command, log_path, prefix: calls.append(command) or True)

    checkpoint = tmp_path / 'best_model.pth'
    checkpoint.write_bytes(b'checkpoint')
    job = {
        'model': SimpleNamespace(id='unet'),
        'seed': 0,
        'gpu': '0',
        'output_root': tmp_path,
        'marker': tmp_path / '.eval_done',
        'command': 'evaluate',
        'checkpoint': checkpoint,
        'eval_artifact': tmp_path / 'eval' / 'tables' / 'all_metrics.csv',
    }
    _write_marker(job)

    # Dry-run reports the rerun but must not delete the stale marker.
    assert _run_job(job, 'eval', resume=True, dry_run=True)
    assert job['marker'].is_file()
    assert len(calls) == 0

    # Stale marker + missing table must rerun eval.
    assert _run_job(job, 'eval', resume=True, dry_run=False)
    assert len(calls) == 1

    # Marker + a non-empty table is a valid eval result.
    artifact = tmp_path / 'eval' / 'tables' / 'all_metrics.csv'
    artifact.parent.mkdir(parents=True)
    artifact.write_text('metric\n')
    assert _run_job(job, 'eval', resume=True, dry_run=False)
    assert len(calls) == 1

    # Deleting the table invalidates the marker again.
    artifact.unlink()
    assert _run_job(job, 'eval', resume=True, dry_run=False)
    assert len(calls) == 2


def test_eval_artifact_paths(tmp_path):
    from types import SimpleNamespace

    assert _eval_artifact(SimpleNamespace(id='unet'), tmp_path) == \
        tmp_path / 'eval' / 'tables' / 'all_metrics.csv'
    assert _eval_artifact(SimpleNamespace(id='skein'), tmp_path) == \
        tmp_path / 'eval' / 'tables' / 'all_metrics.csv'


def test_resolve_models_all_returns_only_available():
    models = load_models()
    resolved = resolve_models('all', models)
    assert [model.id for model in resolved] == sorted(
        model.id for model in models.values() if model.available)


def test_seeded_protocol_expands_five_splits():
    protocol = load_protocol('seeded5')
    assert list(protocol.seeds) == [0, 1, 2, 3, 4]
    split = protocol.split_for(3)
    assert split.split_root.as_posix().endswith('data/sacm_3shot_seed3')
    assert split.train_manifest.name == 'split_manifest.csv'
    assert split.test_manifest.name == 'protocol_manifest.csv'


def test_plan_uses_seed_specific_train_and_test_roots(capsys):
    rc = main(['plan', '--model', 'skein', '--seeds', '0', '1'])
    output = capsys.readouterr().out
    assert rc == 0
    assert 'data/sacm_3shot_seed0' in output
    assert 'data/sacm_3shot_seed1' in output
    assert 'results/runs/skein/skein/seed_0' in output
    assert 'results/runs/skein/skein/seed_1' in output
    assert 'TRAIN_ROOT=data/sacm_3shot_seed0' in output
    assert 'TRAIN_ROOT=data/sacm_3shot_seed1' in output


def test_plan_expands_unet_seed_paths(capsys):
    rc = main(['plan', '--model', 'unet', '--seeds', '0'])
    output = capsys.readouterr().out
    assert rc == 0
    assert '--data-root data/sacm_3shot_seed0' in output
    assert '--protocol data/sacm_3shot_seed0/protocol_manifest.csv' in output
    assert 'results/runs/unet/default/seed_0' in output


def test_train_dry_run_does_not_create_marker(capsys):
    rc = main(['train', '--model', 'skein', '--seeds', '0', '--dry-run'])
    output = capsys.readouterr().out
    assert rc == 0
    assert 'dry-run:' in output
    assert 'TRAIN_ROOT=data/sacm_3shot_seed0' in output


def test_doctor_reports_self_contained_models(capsys):
    main(['doctor'])
    output = capsys.readouterr().out
    assert 'models: 44  available: 44' in output
    assert 'sam2_hiera_large.pt' in output
    assert 'SAM2 Hiera-L download' in output
    assert 'DINOv3' in output or 'segdino' in output


def test_select_cuda_device_targets_job_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.cuda, 'set_device', lambda index: calls.append(index))

    _select_cuda_device(torch.device('cuda:3'))
    assert calls == [3]
    _select_cuda_device(torch.device('cpu'))
    _select_cuda_device(torch.device('cuda'))
    assert calls == [3]


def test_plan_rejects_unknown_seed(capsys):
    rc = main(['plan', '--model', 'skein', '--seeds', '9'])
    assert rc == 2
    assert 'unknown' in capsys.readouterr().err


def test_unet_small_image_padding_keeps_random_crop_margin():
    from baseline_zoo.adapters.unet import UNetAdapter

    adapter = UNetAdapter()
    adapter.patch_size = 512
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    mask = np.zeros((300, 300), dtype=bool)
    padded_image, padded_mask = adapter._pad_to_patch(image, mask)
    margin = max(16, adapter.patch_size // 16)
    expected = adapter.patch_size + 2 * margin
    assert padded_image.shape == (expected, expected, 3)
    assert padded_mask.shape == (expected, expected)
    # The padded canvas must be larger than one patch, otherwise
    # _crop_origin still has nowhere to move.
    assert padded_image.shape[0] > adapter.patch_size
    assert padded_image.shape[1] > adapter.patch_size


def test_unet_rejects_nonpositive_scale():
    from types import SimpleNamespace
    from baseline_zoo.adapters.unet import UNetAdapter

    adapter = UNetAdapter()
    with pytest.raises(ValueError):
        adapter.build_model(SimpleNamespace(scale=0.0))

