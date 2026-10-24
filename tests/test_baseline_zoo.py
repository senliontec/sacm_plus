"""Tests for the model registry, protocol expansion and dry-run planner."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.__main__ import main  # noqa: E402
from baseline_zoo.protocols import load_protocol  # noqa: E402
from baseline_zoo.registry import load_models, resolve_models  # noqa: E402


def test_registry_contains_all_self_contained_models():
    models = load_models()
    expected = {
        'bcunet', 'cs2net', 'cwsam', 'fob_sam', 'knowsam', 'maskvsc',
        'nnwnet', 'sacm_official', 'sacm_x', 'sam2_unet',
        'sam_med2d', 'sam_octa', 'sam_vit_l_unet', 'segdino', 'unet'}
    assert set(models) == expected
    available = {model.id for model in models.values() if model.available}
    assert available == expected - {'sam2_unet'}
    assert models['sam2_unet'].status == 'reference_only'
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
    assert 'models: 15  available: 14' in output
    assert 'sam2_hiera_large.pt' in output
    assert 'DINOv3' in output or 'segdino' in output


def test_plan_rejects_unknown_seed(capsys):
    rc = main(['plan', '--model', 'sacm_x', '--seeds', '9'])
    assert rc == 2
    assert 'unknown' in capsys.readouterr().err
