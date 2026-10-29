"""Tests for the model registry, protocol expansion and dry-run planner."""

import sys
from pathlib import Path

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
        'rsrm', 's2mnet', 'sacm_official', 'sacm_x', 'sam2_unet',
        'sam_adapter',
        'sam_med2d', 'sam_octa', 'sam_vit_l_unet', 'samed', 'sansa',
        'scrwkv', 'segdino', 'segdino_v2', 'topolora', 'topoweave',
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
    assert 'scripts/sacm_x/run_sacm_x_quick.sh' in models['sacm_x'].train_entrypoint
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
    assert 'results/paper/sam_octa/seed_0' in output


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
        'eval_artifact': tmp_path / 'paper_metrics.csv',
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
    (tmp_path / 'paper_metrics.csv').write_text('metric\n')
    assert _run_job(job, 'eval', resume=True, dry_run=False)
    assert len(calls) == 1

    # Deleting the table invalidates the marker again.
    (tmp_path / 'paper_metrics.csv').unlink()
    assert _run_job(job, 'eval', resume=True, dry_run=False)
    assert len(calls) == 2


def test_eval_artifact_paths(tmp_path):
    from types import SimpleNamespace

    assert _eval_artifact(SimpleNamespace(id='unet'), tmp_path) == \
        tmp_path / 'paper_metrics.csv'
    assert _eval_artifact(SimpleNamespace(id='sacm_x'), tmp_path) == \
        tmp_path / 'tables' / 'all_metrics.csv'


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
    rc = main(['plan', '--model', 'sacm_x', '--seeds', '0', '1'])
    output = capsys.readouterr().out
    assert rc == 0
    assert 'data/sacm_3shot_seed0' in output
    assert 'data/sacm_3shot_seed1' in output
    assert 'results/paper/sacm_x/seed_0' in output
    assert 'results/paper/sacm_x/seed_1' in output
    assert 'TRAIN_ROOT=data/sacm_3shot_seed0' in output
    assert 'TRAIN_ROOT=data/sacm_3shot_seed1' in output


def test_plan_expands_unet_seed_paths(capsys):
    rc = main(['plan', '--model', 'unet', '--seeds', '0'])
    output = capsys.readouterr().out
    assert rc == 0
    assert '--data-root data/sacm_3shot_seed0' in output
    assert '--protocol data/sacm_3shot_seed0/protocol_manifest.csv' in output
    assert 'results/paper/unet/seed_0' in output


def test_train_dry_run_does_not_create_marker(capsys):
    rc = main(['train', '--model', 'sacm_x', '--seeds', '0', '--dry-run'])
    output = capsys.readouterr().out
    assert rc == 0
    assert 'dry-run:' in output
    assert 'TRAIN_ROOT=data/sacm_3shot_seed0' in output


def test_doctor_reports_self_contained_models(capsys):
    main(['doctor'])
    output = capsys.readouterr().out
    assert 'models: 43  available: 43' in output
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
    rc = main(['plan', '--model', 'sacm_x', '--seeds', '9'])
    assert rc == 2
    assert 'unknown' in capsys.readouterr().err
