"""Tests for the config-driven SACM-STC B0-B3 ablation plumbing."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))


def _load_module(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_module(
    'sacm_stc_ablation_runner',
    'scripts/pipeline/run_sacm_stc_ablation.py')
ANALYZER = _load_module(
    'sacm_stc_ablation_analyzer',
    'scripts/pipeline/analyze_sacm_stc_ablation.py')
DIAGNOSTICS = _load_module(
    'sacm_stc_candidate_diagnostics',
    'scripts/pipeline/diagnose_sacm_stc_candidates.py')
PAPER_TABLE = _load_module(
    'sacm_stc_paper_table',
    'scripts/pipeline/build_paper_table.py')
CONFIG = REPO_ROOT / 'configs' / 'ablations' / \
    'sacm_stc_scope_structure.yaml'
QUICK_CONFIG = REPO_ROOT / 'configs' / 'ablations' / \
    'sacm_stc_das_quick.yaml'
FINAL_CONFIG = REPO_ROOT / 'configs' / 'ablations' / \
    'sacm_stc_final.yaml'


def test_sacm_stc_default_alias_is_rejected():
    assert PAPER_TABLE._parse_models(['sacm_stc']) == [
        ('sacm_stc', 'all_dense')]
    with pytest.raises(SystemExit):
        PAPER_TABLE._parse_models(['sacm_stc:default'])


def test_ablation_config_and_train_command_parse():
    config = RUNNER.load_config(CONFIG)
    assert list(config['arms']) == [
        'b0_selected_base',
        'b1_all_dense_base',
        'ss_selected_dense_selected_structure',
        'sa_selected_dense_all_structure',
        'b2_all_dense_selected_structure',
        'b3_all_dense_all_structure',
    ]
    assert config['arms']['ss_selected_dense_selected_structure']['args'][
        'topology-head'] == 'selected'
    assert config['arms']['sa_selected_dense_all_structure']['args'][
        'topology-head'] == 'all'
    assert config['arms']['b2_all_dense_selected_structure']['args'][
        'topology-head'] == 'selected'
    assert config['arms']['b3_all_dense_all_structure']['args'][
        'topology-head'] == 'all'

    from baseline_zoo.adapters.registry import load_adapter
    from baseline_zoo.runtime import build_parser

    parser = build_parser(load_adapter('sacm_stc'))
    for arm, scope, topology in (
            ('b0_selected_base', 'selected', 'selected'),
            ('ss_selected_dense_selected_structure', 'selected', 'selected'),
            ('sa_selected_dense_all_structure', 'selected', 'all'),
            ('b2_all_dense_selected_structure', 'all', 'selected'),
            ('b3_all_dense_all_structure', 'all', 'all')):
        command = RUNNER._train_command(
            config, arm, 0, '0', Path('data/sacm_3shot_seed0'),
            Path(f'results/runs/sacm_stc/{arm}/seed_0'), 200, False)
        parsed = parser.parse_args(command[5:])
        assert parsed.stage == 'train'
        assert parsed.stage2_loss_scope == scope
        assert parsed.topology_head == topology
        assert parsed.epochs == 200
        assert parsed.use_wandb is True
        assert parsed.wandb_name.endswith(f'{arm}-seed0')


def test_quick_das_config_and_arm_flags_parse():
    config = RUNNER.load_config(QUICK_CONFIG)
    assert config['eval_checkpoint'] == 'best_model.pth'
    assert config['analysis']['require_train_complete'] is True
    assert config['analysis']['require_eval_complete'] is True
    assert list(config['arms']) == [
        'das_ss',
        'das_aa_mean',
        'das_aa_dep',
        'das_as_dep',
    ]

    from baseline_zoo.adapters.registry import load_adapter
    from baseline_zoo.runtime import build_parser

    parser = build_parser(load_adapter('sacm_stc'))
    cases = {
        'das_ss': ('selected', 'selected', 'deployment'),
        'das_aa_mean': ('all', 'all', 'mean'),
        'das_aa_dep': ('all', 'all', 'deployment'),
        'das_as_dep': ('all', 'selected', 'deployment'),
    }
    for arm, (dense, topology, aggregate) in cases.items():
        command = RUNNER._train_command(
            config, arm, 0, '0', Path('data/sacm_3shot_seed0'),
            Path(f'results/runs/sacm_stc/{arm}/seed_0'), 300, False)
        parsed = parser.parse_args(command[5:])
        assert parsed.stage2_loss_scope == dense
        assert parsed.topology_head == topology
        if arm == 'das_aa_mean':
            assert parsed.stage2_loss_aggregate == 'mean'
            assert parsed.topology_aggregate == 'mean'
        else:
            assert parsed.stage2_loss_aggregate == aggregate
            assert parsed.topology_aggregate == 'deployment'


def test_final_config_is_canonical_and_parses_all_arms():
    config = RUNNER.load_config(FINAL_CONFIG)
    assert config['output_root'] == 'results/runs/sacm_stc'
    assert config['eval_checkpoint'] == 'best_model.pth'
    assert config['epochs'] == 800
    assert config['analysis']['require_train_complete'] is True
    assert config['analysis']['require_eval_complete'] is True
    assert list(config['arms']) == [
        'final_full',
        'final_sel_sel',
        'final_sel_all',
        'final_all_sel',
        'final_all_all_mean',
        'final_sel_none',
        'final_all_none',
    ]

    from baseline_zoo.adapters.registry import load_adapter
    from baseline_zoo.runtime import build_parser

    parser = build_parser(load_adapter('sacm_stc'))
    for arm in config['arms']:
        command = RUNNER._train_command(
            config, arm, 0, '0', Path('data/sacm_3shot_seed0'),
            Path(f'results/runs/sacm_stc/{arm}/seed_0'), 800, False)
        parsed = parser.parse_args(command[5:])
        assert parsed.epochs == 800
        assert parsed.stage2_loss_scope in ('all', 'selected')
        assert parsed.topology_head in ('all', 'selected')
        if arm in ('final_sel_none', 'final_all_none'):
            assert parsed.cl_dice_weight == 0.0
            assert parsed.skeleton_recall_weight == 0.0
        else:
            assert parsed.cl_dice_weight == 0.1
            assert parsed.skeleton_recall_weight == 0.1

    from baseline_zoo.protocols import load_protocol
    protocol = load_protocol('seeded5')
    jobs = RUNNER.build_jobs(
        config, protocol, 'train', list(config['arms']),
        config['seeds'], 800, False)
    assert len(jobs) == 7 * 4
    assert all([spec['stage'] for spec in job['commands']] == ['train']
               for job in jobs)

    assert PAPER_TABLE._model_label('sacm_stc', 'final_full') == \
        'SACM-STC (full)'
    assert PAPER_TABLE._model_label(
        'sacm_stc', 'final_all_all_mean') == \
        'SACM-STC (all/all, uniform control)'


def test_audit_jobs_build_diagnostic_commands():
    config = RUNNER.load_config(CONFIG)
    from baseline_zoo.protocols import load_protocol
    protocol = load_protocol('seeded5')
    jobs = RUNNER.build_jobs(
        config, protocol, 'audit', [], [0, 1], 200, False)
    assert len(jobs) == 2
    for job, seed in zip(jobs, (0, 1)):
        command = job['commands'][0]['command']
        assert 'diagnose_sacm_stc_candidates.py' in command[1]
        assert f'seed_{seed}/best_model.pth' in command[command.index(
            '--checkpoint') + 1]
        assert command[command.index('--max-per-dataset') + 1] == '20'


def test_full_matrix_has_24_jobs_and_each_has_three_stages():
    config = RUNNER.load_config(CONFIG)
    from baseline_zoo.protocols import load_protocol
    protocol = load_protocol('seeded5')
    jobs = RUNNER.build_jobs(
        config, protocol, 'all', list(config['arms']),
        config['seeds'], 200, False)
    assert len(jobs) == 6 * 4
    assert all([spec['stage'] for spec in job['commands']]
               == ['train', 'eval', 'diagnose'] for job in jobs)


def test_eval_completion_requires_all_datasets(tmp_path):
    run = tmp_path / 'run'
    table = run / 'eval' / 'tables' / 'all_metrics.csv'
    table.parent.mkdir(parents=True)
    with open(table, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset'])
        writer.writeheader()
        writer.writerows([{'dataset': f'D{i}'} for i in range(20)])
    assert RUNNER._eval_complete(run, 20) is True
    assert RUNNER._eval_complete(run, 21) is False
    with open(table, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset'])
        writer.writeheader()
        writer.writerows([{'dataset': f'D{i}'} for i in range(19)])
    assert RUNNER._eval_complete(run, 20) is False
    manifest = run / 'eval' / 'run_manifest.json'
    manifest.write_text('{"status": "partial"}')
    assert RUNNER._eval_complete(run, 19) is False


def test_resume_does_not_reuse_eval_when_training_will_rerun(tmp_path):
    import copy

    config = copy.deepcopy(RUNNER.load_config(CONFIG))
    config['output_root'] = str(tmp_path / 'runs')
    config['diagnostics']['output_root'] = str(tmp_path / 'diagnostics')
    from baseline_zoo.protocols import load_protocol
    protocol = load_protocol('seeded5')
    arm = 'b0_selected_base'
    run = tmp_path / 'runs' / arm / 'seed_0'
    table = run / 'eval' / 'tables' / 'all_metrics.csv'
    _write_csv(table, [{'dataset': f'D{i}'} for i in range(20)],
               ['dataset'])
    (run / 'eval').mkdir(parents=True, exist_ok=True)
    (run / 'eval' / 'run_manifest.json').write_text(
        '{"status": "done", "checkpoint": "'
        + str(run / 'last_model.pth') + '"}')

    jobs = RUNNER.build_jobs(
        config, protocol, 'all', [arm], [0], 200, True)
    commands = jobs[0]['commands']
    # No run_info => training runs; the old complete eval must not be reused.
    assert commands[0]['skip'] is None
    assert commands[1]['skip'] is None
    assert commands[2]['skip'] is None

    (run / 'run_info.json').write_text(
        '{"status": "done", "epochs": 200}')
    (tmp_path / 'diagnostics' / arm / 'seed_0').mkdir(parents=True)
    (tmp_path / 'diagnostics' / arm / 'seed_0' / 'summary.json').write_text(
        '{"checkpoint": "' + str(run / 'last_model.pth') + '"}')
    jobs = RUNNER.build_jobs(
        config, protocol, 'all', [arm], [0], 200, True)
    commands = jobs[0]['commands']
    assert commands[0]['skip'] is not None
    assert commands[1]['skip'] is not None
    assert commands[2]['skip'] is not None

    # A partial old eval table must also be rerun.
    _write_csv(table, [{'dataset': f'D{i}'} for i in range(19)],
               ['dataset'])
    jobs = RUNNER.build_jobs(
        config, protocol, 'all', [arm], [0], 200, True)
    commands = jobs[0]['commands']
    assert commands[0]['skip'] is not None
    assert commands[1]['skip'] is None
    assert commands[2]['skip'] is None


def test_resume_requires_matching_epoch_budget(tmp_path):
    import copy

    config = copy.deepcopy(RUNNER.load_config(CONFIG))
    config['output_root'] = str(tmp_path / 'runs')
    config['diagnostics']['output_root'] = str(tmp_path / 'diagnostics')
    from baseline_zoo.protocols import load_protocol
    protocol = load_protocol('seeded5')
    arm = 'b0_selected_base'
    run = tmp_path / 'runs' / arm / 'seed_0'
    run.mkdir(parents=True)
    (run / 'run_info.json').write_text(
        '{"status": "done", "epochs": 200}')

    # A completed 200-epoch run must not satisfy a 300-epoch request.
    jobs = RUNNER.build_jobs(
        config, protocol, 'all', [arm], [0], 300, True)
    assert jobs[0]['commands'][0]['skip'] is None

    # The same run is reusable when the budget matches.
    jobs = RUNNER.build_jobs(
        config, protocol, 'all', [arm], [0], 200, True)
    assert jobs[0]['commands'][0]['skip'] is not None


def test_resume_rejects_partial_run_with_other_epoch_budget(tmp_path):
    import copy

    config = copy.deepcopy(RUNNER.load_config(CONFIG))
    config['output_root'] = str(tmp_path / 'runs')
    config['diagnostics']['output_root'] = str(tmp_path / 'diagnostics')
    from baseline_zoo.protocols import load_protocol
    protocol = load_protocol('seeded5')
    arm = 'b1_all_dense_base'
    run = tmp_path / 'runs' / arm / 'seed_1'
    run.mkdir(parents=True)
    (run / 'last_model.pth').write_bytes(b'partial')
    (run / 'train_config.json').write_text(
        '{"args": {"epochs": 200}}')

    jobs = RUNNER.build_jobs(
        config, protocol, 'all', [arm], [1], 300, True)
    assert '--resume' not in jobs[0]['commands'][0]['command']

    jobs = RUNNER.build_jobs(
        config, protocol, 'all', [arm], [1], 200, True)
    assert '--resume' in jobs[0]['commands'][0]['command']


def test_eval_stage_requires_completed_training(tmp_path):
    import copy

    config = copy.deepcopy(RUNNER.load_config(CONFIG))
    config['output_root'] = str(tmp_path / 'runs')
    config['diagnostics']['output_root'] = str(tmp_path / 'diagnostics')
    from baseline_zoo.protocols import load_protocol
    protocol = load_protocol('seeded5')
    arm = 'b0_selected_base'
    run = tmp_path / 'runs' / arm / 'seed_0'

    jobs = RUNNER.build_jobs(
        config, protocol, 'eval', [arm], [0], 300, True)
    assert jobs[0]['commands'] == []
    assert 'run_info' in jobs[0]['error']

    run.mkdir(parents=True)
    info = run / 'run_info.json'
    info.write_text('{"status": "running", "epochs": 300}')
    jobs = RUNNER.build_jobs(
        config, protocol, 'eval', [arm], [0], 300, True)
    assert 'status' in jobs[0]['error']

    info.write_text('{"status": "done", "epochs": 200}')
    jobs = RUNNER.build_jobs(
        config, protocol, 'eval', [arm], [0], 300, True)
    assert 'epochs' in jobs[0]['error']

    info.write_text('{"status": "done", "epochs": 300}')
    jobs = RUNNER.build_jobs(
        config, protocol, 'eval', [arm], [0], 300, True)
    assert 'error' not in jobs[0]
    assert [spec['stage'] for spec in jobs[0]['commands']] == ['eval']

    # Explicit override remains available for externally produced checkpoints.
    info.unlink()
    jobs = RUNNER.build_jobs(
        config, protocol, 'eval', [arm], [0], 300, True,
        allow_untrained_eval=True)
    assert 'error' not in jobs[0]
    assert [spec['stage'] for spec in jobs[0]['commands']] == ['eval']


def test_analysis_can_require_completed_training(tmp_path):
    base = tmp_path / 'runs'
    arm = 'arm_a'
    run = base / arm / 'seed_0'
    tables = run / 'eval' / 'tables'
    tables.mkdir(parents=True)
    macro = []
    for role in ('All', 'Seen', 'Unseen', 'Novel'):
        for metric in ('dice', 'iou', 'cldice', 'hd95'):
            value = 50.0 if metric == 'hd95' else 0.5
            macro.append({'role': role, 'metric': metric, 'mean': value})
    _write_csv(tables / 'macro_metrics.csv', macro,
               ['role', 'metric', 'mean'])
    _write_csv(tables / 'all_metrics.csv', [{
        'dataset': 'D', 'role': 'Seen',
        'dice_mean': 0.5, 'iou_mean': 0.5,
        'cldice_mean': 0.5, 'hd95_mean': 50.0,
    }], ['dataset', 'role', 'dice_mean', 'iou_mean',
         'cldice_mean', 'hd95_mean'])
    (run / 'eval' / 'run_manifest.json').write_text(
        '{"status": "done", "checkpoint": "/x/best_model.pth"}')

    config = {
        'schema_version': 1,
        'name': 'require_complete_test',
        'protocol': 'seeded5',
        'method': 'sacm_stc',
        'seeds': [0],
        'epochs': 300,
        'expected_datasets': 1,
        'eval_checkpoint': 'best_model.pth',
        'output_root': str(base),
        'arms': {arm: {'args': {}}},
        'contrasts': [],
        'analysis': {
            'output_root': str(tmp_path / 'analysis'),
            'require_train_complete': True,
            'require_eval_complete': True,
            'bootstrap_samples': 10,
            'seed': 1,
        },
        'diagnostics': {'output_root': str(tmp_path / 'diagnostics')},
    }
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))

    # No contrast is configured, so the analyzer's exit code is 2; the
    # assertion here is about what was admitted to the tables.
    assert ANALYZER.main(['--config', str(config_path)]) in (0, 2)
    summary = json.loads((tmp_path / 'analysis' / 'summary.json').read_text())
    assert int(summary['n_arm_rows']) == 0
    assert any('run_info' in item for item in summary['missing'])

    run.mkdir(parents=True, exist_ok=True)
    (run / 'run_info.json').write_text(
        '{"status": "done", "epochs": 300}')
    assert ANALYZER.main(['--config', str(config_path)]) in (0, 2)
    summary = json.loads((tmp_path / 'analysis' / 'summary.json').read_text())
    assert int(summary['n_arm_rows']) > 0
    assert not any('run_info' in item for item in summary['missing'])

    # A partial eval table must be rejected even when training is complete.
    _write_csv(tables / 'all_metrics.csv', [],
               ['dataset', 'role', 'dice_mean', 'iou_mean',
                'cldice_mean', 'hd95_mean'])
    assert ANALYZER.main(['--config', str(config_path)]) in (0, 2)
    summary = json.loads((tmp_path / 'analysis' / 'summary.json').read_text())
    assert int(summary['n_arm_rows']) == 0
    assert any('eval table' in item for item in summary['missing'])


def test_matrix_evaluates_the_configured_final_checkpoint():
    config = RUNNER.load_config(CONFIG)
    assert config['eval_checkpoint'] == 'last_model.pth'
    from baseline_zoo.protocols import load_protocol
    protocol = load_protocol('seeded5')
    jobs = RUNNER.build_jobs(
        config, protocol, 'all', ['b0_selected_base'], [0], 300, True)
    commands = jobs[0]['commands']
    eval_command = commands[1]['command']
    diagnose_command = commands[2]['command']
    assert 'last_model.pth' in eval_command[
        eval_command.index('--checkpoint') + 1]
    assert 'last_model.pth' in diagnose_command[
        diagnose_command.index('--checkpoint') + 1]


def test_diagnostic_helpers_are_deterministic():
    rows = ([{'dataset': 'A', 'source_image': str(i)} for i in range(10)]
            + [{'dataset': 'B', 'source_image': str(i)} for i in range(3)])
    selected = DIAGNOSTICS._stratified_limit(rows, 4)
    assert [row['source_image'] for row in selected
            if row['dataset'] == 'A'] == ['0', '3', '6', '9']
    assert sum(row['dataset'] == 'B' for row in selected) == 3
    assert DIAGNOSTICS._candidate_indices([0.5, 0.7, 0.6, 0.4], 0) == (1, 3)
    # Ties favor the deployed slot so gating correctness is not penalised
    # by an arbitrary argmax order.
    assert DIAGNOSTICS._candidate_indices([0.7, 0.7, 0.6, 0.4], 0) == (0, 3)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_analysis_paired_contrasts_from_canonical_tables(tmp_path):
    arms = [
        'b0_selected_base',
        'b1_all_dense_base',
        'b2_all_dense_selected_structure',
        'b3_all_dense_all_structure',
    ]
    base = tmp_path / 'runs'
    offsets = {
        'b0_selected_base': 0.50,
        'b1_all_dense_base': 0.51,
        'b2_all_dense_selected_structure': 0.52,
        'b3_all_dense_all_structure': 0.53,
    }
    diag_root = tmp_path / 'diagnostics'
    for arm in arms:
        for seed in (0, 1):
            run = base / arm / f'seed_{seed}' / 'eval' / 'tables'
            run.mkdir(parents=True)
            dice = offsets[arm] + 0.001 * seed
            macro = []
            for role in ('All', 'Seen', 'Unseen', 'Novel'):
                for metric in ('dice', 'iou', 'cldice', 'hd95'):
                    value = dice if metric == 'dice' else (
                        0.4 if metric != 'hd95' else 50.0)
                    macro.append({
                        'role': role, 'metric': metric, 'mean': value})
            _write_csv(run / 'macro_metrics.csv', macro,
                       ['role', 'metric', 'mean'])
            per_dataset = []
            for dataset, role in (('A', 'Seen'), ('B', 'Unseen')):
                per_dataset.append({
                    'dataset': dataset, 'role': role,
                    'dice_mean': dice, 'iou_mean': 0.4,
                    'cldice_mean': 0.4, 'hd95_mean': 50.0,
                })
            _write_csv(run / 'all_metrics.csv', per_dataset,
                       ['dataset', 'role', 'dice_mean', 'iou_mean',
                        'cldice_mean', 'hd95_mean'])
            diag = diag_root / arm / f'seed_{seed}'
            diag.mkdir(parents=True)
            _write_csv(diag / 'selection_by_role.csv', [{
                'split': 'test', 'role': 'All',
                'selected_dice_mean': dice,
                'oracle_dice_mean': dice + 0.01,
                'gating_top1_mean': 1.0,
                'dice_regret_mean': 0.01,
                'pairwise_disagreement_mean': 0.1,
            }], ['split', 'role', 'selected_dice_mean',
                 'oracle_dice_mean', 'gating_top1_mean',
                 'dice_regret_mean', 'pairwise_disagreement_mean'])

    config = {
        'schema_version': 1,
        'name': 'synthetic_ablation',
        'protocol': 'seeded5',
        'method': 'sacm_stc',
        'seeds': [0, 1],
        'output_root': str(base),
        'arms': {arm: {'args': {}} for arm in arms},
        'contrasts': [
            {'id': 'b1_minus_b0', 'left': 'b1_all_dense_base',
             'right': 'b0_selected_base'},
            {'id': 'b3_minus_b2',
             'left': 'b3_all_dense_all_structure',
             'right': 'b2_all_dense_selected_structure'},
        ],
        'analysis': {
            'output_root': str(tmp_path / 'analysis'),
            'primary_metric': 'dice',
            'primary_role': 'All',
            'bootstrap_samples': 100,
            'seed': 1,
        },
        'diagnostics': {'output_root': str(tmp_path / 'diagnostics')},
    }
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    code = ANALYZER.main(['--config', str(config_path)])
    assert code == 0
    rows = list(csv.DictReader(
        open(tmp_path / 'analysis' / 'contrast_summary.csv')))
    b3_b2 = [row for row in rows
             if row['contrast'] == 'b3_minus_b2'
             and row['role'] == 'All' and row['metric'] == 'dice'][0]
    assert float(b3_b2['mean_diff']) == pytest.approx(0.01)
    assert int(b3_b2['n']) == 2
    assert int(b3_b2['win']) == 2
    arm_rows = list(csv.DictReader(
        open(tmp_path / 'analysis' / 'arm_summary.csv')))
    assert any(row['arm'] == 'b3_all_dense_all_structure'
               and row['role'] == 'All'
               and row['metric'] == 'dice'
               and int(row['n_seeds']) == 2
               for row in arm_rows)
    candidate_rows = list(csv.DictReader(
        open(tmp_path / 'analysis' / 'candidate_contrast_summary.csv')))
    candidate = [row for row in candidate_rows
                 if row['contrast'] == 'b3_minus_b2'
                 and row['role'] == 'All'
                 and row['key'] == 'selected_dice'][0]
    assert float(candidate['mean_diff']) == pytest.approx(0.01)
    report = (tmp_path / 'analysis' / 'ablation_report.md').read_text()
    assert 'b3_minus_b2' in report
    assert 'Primary' in report
    assert 'Candidate-head diagnostics' in report
