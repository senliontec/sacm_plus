"""Freeze checks for the canonical DCD fixed3 800 recipe.

These tests guard the release state:
* the 4-arm runner's ``ssh`` arm is the same recipe as the earlier
  fixed3 800 canonical pair;
* the three ablation arms keep their intended factors;
* the DCD adapter/package no longer exposes the retired structural pilot.
"""

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from baseline_zoo.adapters.dcd import DCDAdapter  # noqa: E402
import baseline_zoo.models.dcd as dcd_models  # noqa: E402


def _load(name: str) -> dict:
    path = REPO_ROOT / 'configs' / 'ablations' / name
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _merged(config: dict, arm: str) -> dict:
    merged = {}
    for mapping in (config.get('common_train_args') or {},
                    (config.get('arms') or {}).get(arm, {}).get('args') or {}):
        for key, value in mapping.items():
            merged[str(key).replace('_', '-')] = value
    return merged


def test_full_800_ssh_is_the_previous_fixed3_800_recipe():
    new = _load('dcd_full_800.yaml')
    old = _load('dcd_ssh_full_ab.yaml')
    new_ssh = _merged(new, 'ssh')
    old_ssh = _merged(old, 'ssh')

    # The new config spells out the adapter default; it does not change it.
    assert new_ssh.pop('adapter-dim-ratio') == 0.1
    assert new_ssh == old_ssh
    assert int(new['epochs']) == int(old['epochs']) == 800
    assert new['seeds'] == old['seeds'] == [0, 1, 2, 3]
    assert new['protocol'] == old['protocol'] == 'seeded5'
    assert 'structural-mode' not in new_ssh


def test_full_800_ablation_factors_are_frozen():
    config = _load('dcd_full_800.yaml')
    nofusion = _merged(config, 'nofusion')
    native = _merged(config, 'native_control')
    notopo = _merged(config, 'notopo')

    assert int(nofusion['specialized-heads']) == 1
    assert int(nofusion['head-fusion']) == 0

    assert int(native['specialized-heads']) == 0
    assert int(native['head-fusion']) == 0

    assert int(notopo['specialized-heads']) == 1
    assert int(notopo['head-fusion']) == 1
    assert float(notopo['cl-dice-weight']) == 0.0
    assert float(notopo['skeleton-recall-weight']) == 0.0
    assert float(notopo['specialized-topology-weight']) == 0.0


def test_dcd_adapter_has_no_structural_pilot_arguments():
    adapter = DCDAdapter()
    parser = argparse.ArgumentParser()
    adapter.add_args(parser)
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert '--structural-mode' not in options
    assert not any('structural' in option for option in options)
    assert not hasattr(adapter, 'structural_mode')
    assert dcd_models.__all__ == [
        'build_sam_vit_b',
        'build_sam_vit_h',
        'build_sam_vit_l',
    ]


def test_model_registry_defaults_to_fixed3_800_recipe():
    config = yaml.safe_load(
        (REPO_ROOT / 'configs' / 'models' / 'dcd.yaml').read_text(
            encoding='utf-8')) or {}
    assert int(config['defaults']['epochs']) == 800
    train = str(config['train_entrypoint'])
    for token in (
            '--head-select gating',
            '--head-fusion 1',
            '--specialized-heads 1',
            '--deployment-head 3',
            'make_fixed3_checkpoint.py'):
        assert token in train
    assert 'structural' not in train.lower()
