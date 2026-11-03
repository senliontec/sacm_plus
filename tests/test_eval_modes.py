"""Tests for the optional multi-dataset and paper-only evaluation paths."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def evaluate_mod():
    return _load_module('sacm_evaluate_under_test', 'src/eval/evaluate.py')


@pytest.fixture(scope='module')
def launcher_mod():
    return _load_module('sacm_launcher_under_test',
                        'scripts/sacm_x/run_experiments.py')


def test_parse_test_datasets_validates_names(evaluate_mod):
    specs = evaluate_mod._parse_test_datasets(
        ['DRIVE:/data/DRIVE', 'WIRE:/data/WIRE'])
    assert specs == [('DRIVE', '/data/DRIVE'), ('WIRE', '/data/WIRE')]

    for bad in (['DRIVE'], ['DRIVE:'], [':/data/DRIVE'],
                ['../DRIVE:/data/DRIVE']):
        with pytest.raises(ValueError):
            evaluate_mod._parse_test_datasets(bad)
    with pytest.raises(ValueError):
        evaluate_mod._parse_test_datasets(
            ['DRIVE:/a', 'DRIVE:/b'])


def test_paper_metric_keys_are_the_four_main_metrics(evaluate_mod):
    paper = SimpleNamespace(
        paper_metrics_only=True, auc=True,
        betti_matching=False, topograph_metric=False)
    assert evaluate_mod._resolve_metric_keys(paper) == [
        'dice', 'iou', 'cldice', 'hd95']

    full = SimpleNamespace(
        paper_metrics_only=False, auc=False,
        betti_matching=False, topograph_metric=False)
    assert evaluate_mod._resolve_metric_keys(full) == evaluate_mod.METRIC_KEYS


def test_evaluate_many_reuses_model_and_isolates_failure(
        monkeypatch, tmp_path, evaluate_mod):
    calls = {'prepare': 0, 'run': []}

    def fake_prepare(args):
        calls['prepare'] += 1
        return {'metric_keys': ['dice', 'iou', 'cldice', 'hd95']}

    def fake_run(ctx, args):
        calls['run'].append(args.data_root)
        if args.data_root.endswith('/bad'):
            raise RuntimeError('synthetic failure')
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / 'results_summary.txt').write_text('ok\n')

    monkeypatch.setattr(evaluate_mod, 'setup_logging', lambda: None)
    monkeypatch.setattr(evaluate_mod, '_prepare_evaluation', fake_prepare)
    monkeypatch.setattr(evaluate_mod, '_run_dataset', fake_run)
    monkeypatch.setattr(
        evaluate_mod.torch.cuda, 'empty_cache', lambda: None, raising=False)

    out_dir = tmp_path / 'eval'
    args = SimpleNamespace(
        test_datasets=['GOOD:/data/good', 'BAD:/data/bad'],
        output_dir=str(out_dir),
        trained_weights=str(tmp_path / 'best_model.pth'),
    )
    with pytest.raises(SystemExit):
        evaluate_mod.evaluate_many(args)

    assert calls['prepare'] == 1
    assert calls['run'] == ['/data/good', '/data/bad']
    payload = json.loads(
        (out_dir / 'multi_eval_status.json').read_text())
    assert payload['datasets']['GOOD']['status'] == 'done'
    assert payload['datasets']['BAD']['status'] == 'failed'
    assert (out_dir / 'BAD' / 'eval_error.txt').is_file()


def test_load_multi_eval_status(launcher_mod, tmp_path):
    preset_dir = tmp_path / 'eval' / 'full'
    preset_dir.mkdir(parents=True)
    (preset_dir / 'multi_eval_status.json').write_text(json.dumps({
        'datasets': {
            'DRIVE': {'status': 'done'},
            'WIRE': {'status': 'failed'},
        },
    }))
    args = SimpleNamespace(output_root=str(tmp_path / 'eval'))
    statuses = launcher_mod.load_multi_eval_status(args, 'full')
    assert statuses['DRIVE']['status'] == 'done'
    assert statuses['WIRE']['status'] == 'failed'
    assert launcher_mod.load_multi_eval_status(args, 'missing') == {}


def test_reused_test_command_uses_test_datasets(launcher_mod, tmp_path):
    args = SimpleNamespace(
        output_root=str(tmp_path / 'eval'),
        pred_threshold=0.0,
        eval_input_mode='aspect_pad',
        seed=0,
        num_workers=4,
        deterministic=True,
        deterministic_strict=True,
        fixed_head=None,
        empty_guard=True,
        tta=False,
        paper_metrics_only=True,
        extra_eval_args='',
    )
    cmd = launcher_mod.reused_test_command(
        args, 'full', [('DRIVE', '/data/DRIVE'), ('WIRE', '/data/WIRE')],
        '/checkpoints/best_model.pth', 'iou')
    assert '--data_root' not in cmd
    assert '--test_datasets' in cmd
    assert 'DRIVE:/data/DRIVE' in cmd
    assert 'WIRE:/data/WIRE' in cmd
    assert '--paper_metrics_only' in cmd
    assert cmd[cmd.index('--output_dir') + 1] == str(tmp_path / 'eval' / 'full')
