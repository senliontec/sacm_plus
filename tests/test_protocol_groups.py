"""Tests for protocol-wide subject/volume/batch isolation checks."""

import importlib.util
import os

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts',
                       'build_protocol_manifest.py')
_spec = importlib.util.spec_from_file_location('build_protocol_manifest',
                                              _SCRIPT)
bpm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpm)


def _row(dataset, split, group):
    return {
        'role': 'seen',
        'dataset': dataset,
        'split': split,
        'group': group,
    }


def test_subject_split_is_valid_and_policy_recorded():
    rows = [
        _row('CHASEDB1', 'train', 'CHASEDB1/01'),
        _row('CHASEDB1', 'train', 'CHASEDB1/02'),
        _row('CHASEDB1', 'train', 'CHASEDB1/03'),
        _row('CHASEDB1', 'val', 'CHASEDB1/04'),
        _row('CHASEDB1', 'seen_test', 'CHASEDB1/05'),
    ]
    checks = bpm.validate_group_policy(rows)
    assert checks['ok']
    assert checks['group_policy']['CHASEDB1'] == 'subject'
    assert checks['cross_split_groups'] == []
    assert checks['unexpected_train_duplicate_groups'] == []


def test_cross_split_group_is_rejected():
    rows = [
        _row('CHASEDB1', 'train', 'CHASEDB1/01'),
        _row('CHASEDB1', 'seen_test', 'CHASEDB1/01'),
    ]
    checks = bpm.validate_group_policy(rows)
    assert not checks['ok']
    assert checks['cross_split_groups'] == [
        {'dataset': 'CHASEDB1', 'group': 'CHASEDB1/01',
         'splits': ['seen_test', 'train']},
    ]


def test_chasedb1_train_duplicate_is_rejected():
    rows = [
        _row('CHASEDB1', 'train', 'CHASEDB1/06'),
        _row('CHASEDB1', 'train', 'CHASEDB1/06'),
        _row('CHASEDB1', 'val', 'CHASEDB1/10'),
    ]
    checks = bpm.validate_group_policy(rows)
    assert not checks['ok']
    assert checks['unexpected_train_duplicate_groups'] == [
        {'dataset': 'CHASEDB1', 'group': 'CHASEDB1/06', 'count': 2},
    ]


def test_cremi_volume_repeat_is_allowed():
    rows = [
        _row('CREMI', 'train', 'CREMI/B'),
        _row('CREMI', 'train', 'CREMI/B'),
        _row('CREMI', 'train', 'CREMI/B'),
        _row('CREMI', 'val', 'CREMI/A'),
        _row('CREMI', 'seen_test', 'CREMI/C'),
    ]
    checks = bpm.validate_group_policy(rows)
    assert checks['ok']
    assert checks['train_duplicate_groups'] == [
        {'dataset': 'CREMI', 'group': 'CREMI/B', 'count': 3},
    ]
    assert checks['unexpected_train_duplicate_groups'] == []
    assert 'CREMI' in checks['allowed_exceptions']
