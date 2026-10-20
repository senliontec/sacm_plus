"""Unit tests for patient/volume-grouped few-shot split selection."""

import importlib.util
import os
import random

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts',
                       'prepare_data.py')
_spec = importlib.util.spec_from_file_location('prepare_data', _SCRIPT)
pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pd)


def _pairs(names):
    return [(f'/src/{n}.jpg', f'/src/{n}.png') for n in names]


def test_parse_group_regexes():
    assert pd.parse_group_regexes(['CHASEDB1:Image_(\\d+)[LR]',
                                   'CREMI:sample_([ABC])']) == {
        'CHASEDB1': 'Image_(\\d+)[LR]',
        'CREMI': 'sample_([ABC])',
    }
    with pytest.raises(ValueError):
        pd.parse_group_regexes(['bad-entry'])


def test_group_pairs_by_patient():
    pairs = _pairs(['Image_01L', 'Image_01R', 'Image_02L', 'Image_02R'])
    groups = pd.group_pairs(pairs, 'CHASEDB1', r'Image_(\d+)[LR]')
    assert sorted(groups) == ['CHASEDB1/01', 'CHASEDB1/02']
    assert len(groups['CHASEDB1/01']) == 2

    with pytest.raises(ValueError, match='did not match'):
        pd.group_pairs(_pairs(['nope']), 'CHASEDB1', r'Image_(\d+)[LR]')


def test_select_group_split_is_disjoint():
    pairs = _pairs(['Image_01L', 'Image_01R', 'Image_02L', 'Image_02R',
                    'Image_03L', 'Image_03R'])
    groups = pd.group_pairs(pairs, 'CHASEDB1', r'Image_(\d+)[LR]')
    random.seed(42)
    train, val = pd.select_group_split(groups, shots=3, val_shots=1)

    assert len(train) == 3
    assert len(val) == 1
    assert {g for _, _, g in train}.isdisjoint({g for _, _, g in val})
    train_images = {img for img, _, _ in train}
    assert not (train_images & {img for img, _, _ in val})


def test_select_group_split_rejects_too_few_groups():
    groups = pd.group_pairs(_pairs(['Image_01L', 'Image_01R']),
                            'CHASEDB1', r'Image_(\d+)[LR]')
    with pytest.raises(ValueError, match='not enough disjoint groups'):
        pd.select_group_split(groups, shots=3, val_shots=1)
