"""Unit tests for patient/volume-grouped few-shot split selection."""

import csv
import importlib.util
import os
import random
import sys

import pytest
from PIL import Image

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


def test_select_group_split_uses_distinct_subjects_when_available():
    names = [f'Image_{i:02d}{side}' for i in range(1, 7) for side in 'LR']
    groups = pd.group_pairs(_pairs(names), 'CHASEDB1', r'Image_(\d+)[LR]')
    random.seed(42)
    train, val = pd.select_group_split(groups, shots=3, val_shots=1)

    assert len(train) == 3
    assert len({g for _, _, g in train}) == 3
    assert len(val) == 1
    assert {g for _, _, g in train}.isdisjoint({g for _, _, g in val})


def test_select_group_split_keeps_volume_fallback():
    names = ['sample_A_z000', 'sample_A_z001', 'sample_B_z000',
             'sample_B_z001', 'sample_B_z002', 'sample_C_z000']
    groups = pd.group_pairs(_pairs(names), 'CREMI', r'sample_([ABC])')
    random.seed(42)
    train, val = pd.select_group_split(groups, shots=3, val_shots=1)

    assert len(train) == 3
    assert len({g for _, _, g in train}) == 1
    assert len(val) == 1
    assert {g for _, _, g in train}.isdisjoint({g for _, _, g in val})


def test_preferred_group_order_keeps_cremi_volumes():
    names = ['sample_A_z000', 'sample_B_z000', 'sample_B_z001',
             'sample_B_z002', 'sample_C_z000']
    groups = pd.group_pairs(_pairs(names), 'CREMI', r'sample_([ABC])')
    train, val = pd.select_group_split(
        groups, shots=3, val_shots=1,
        preferred_order=('CREMI/B', 'CREMI/A', 'CREMI/C'))

    assert {g for _, _, g in train} == {'CREMI/B'}
    assert {g for _, _, g in val} == {'CREMI/A'}


def test_sample_within_group_varies_cremi_slices():
    names = ([f'sample_B_z{i:03d}' for i in range(8)]
             + [f'sample_A_z{i:03d}' for i in range(8)]
             + ['sample_C_z000'])
    groups = pd.group_pairs(_pairs(names), 'CREMI', r'sample_([ABC])')
    selections = []
    for seed in (0, 1):
        random.seed(seed)
        train, val = pd.select_group_split(
            groups, shots=3, val_shots=1,
            preferred_order=('CREMI/B', 'CREMI/A', 'CREMI/C'),
            sample_within_group=True)

        assert len(train) == 3
        assert len({img for img, _, _ in train}) == 3
        assert {g for _, _, g in train} == {'CREMI/B'}
        assert {g for _, _, g in val} == {'CREMI/A'}
        selections.append(
            tuple(sorted(os.path.basename(img) for img, _, _ in train)))

    assert selections[0] != selections[1]


def test_legacy_group_split_default_preserves_file_order():
    names = ['sample_B_z000', 'sample_B_z001', 'sample_B_z002',
             'sample_A_z000', 'sample_C_z000']
    groups = pd.group_pairs(_pairs(names), 'CREMI', r'sample_([ABC])')
    random.seed(0)
    train, val = pd.select_group_split(
        groups, shots=3, val_shots=1,
        preferred_order=('CREMI/B', 'CREMI/A', 'CREMI/C'))

    assert [os.path.splitext(os.path.basename(img))[0]
            for img, _, _ in train] == [
        'sample_B_z000', 'sample_B_z001', 'sample_B_z002']
    assert [os.path.splitext(os.path.basename(img))[0]
            for img, _, _ in val] == ['sample_A_z000']


def test_select_group_split_rejects_too_few_groups():
    groups = pd.group_pairs(_pairs(['Image_01L', 'Image_01R']),
                            'CHASEDB1', r'Image_(\d+)[LR]')
    with pytest.raises(ValueError, match='not enough disjoint groups'):
        pd.select_group_split(groups, shots=3, val_shots=1)


def test_parse_named_dirs():
    assert pd.parse_named_dirs(
        ['/data/DRIVE', 'CORN:/data/CORN_train']
    ) == {'DRIVE': '/data/DRIVE', 'CORN': '/data/CORN_train'}
    with pytest.raises(ValueError, match='duplicate'):
        pd.parse_named_dirs(['CORN:/a', 'CORN:/b'])
    with pytest.raises(ValueError, match='invalid'):
        pd.parse_named_dirs(['CORN:'])


def _make_pair(root, stem):
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'masks').mkdir(parents=True, exist_ok=True)
    Image.new('L', (8, 8), 0).save(root / 'images' / f'{stem}.png')
    Image.new('L', (8, 8), 255).save(root / 'masks' / f'{stem}.png')


def test_official_val_and_test_override(tmp_path, monkeypatch):
    train = tmp_path / 'CORN_train'
    val = tmp_path / 'CORN_val'
    test = tmp_path / 'CORN_test'
    out = tmp_path / 'split'
    for i in range(1, 5):
        _make_pair(train, f'batch_{i}_1')
    for i in range(1, 3):
        _make_pair(val, f'batch_9_{i}')
    for i in range(1, 3):
        _make_pair(test, f'batch_10_{i}')

    monkeypatch.setattr(sys, 'argv', [
        'prepare_data.py',
        '--train_dirs', f'CORN:{train}',
        '--val_dirs', f'CORN:{val}',
        '--test_dirs', f'CORN:{test}',
        '--shots', '3', '--val_shots', '1',
        '--make_test', '--seed', '42',
        '--out_dir', str(out),
    ])
    pd.main()

    train_images = sorted((out / 'train' / 'images').glob('*.png'))
    val_images = sorted((out / 'val' / 'images').glob('*.png'))
    test_images = sorted((out / 'test' / 'CORN' / 'images').glob('*.png'))
    assert len(train_images) == 3
    assert len(val_images) == 1
    assert len(test_images) == 2
    assert all('batch_10_' in p.name for p in test_images)
    assert 'batch_9_' in val_images[0].name

    with open(out / 'split_manifest.csv') as f:
        rows = list(csv.DictReader(f))
    assert {r['split'] for r in rows} == {'train', 'val', 'test'}
    assert sum(r['split'] == 'test' for r in rows) == 2
