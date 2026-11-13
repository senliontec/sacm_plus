#!/usr/bin/env python3
"""Fail fast when an existing SSH run was trained with a different recipe.

The fixed3 run reuses ``results/runs/sacm_gcd_ssh_full_ab/ssh/seed_<s>`` when
the ablation runner sees a complete 800-epoch ``run_info.json``.  That skip
predicate checks status and epoch budget only.  This script additionally
compares the existing checkpoint's adapter config against the YAML arm, so a
complete run from a different SSH recipe cannot silently become the fixed3
base.

It is read-only and is intended to run before training:

    python scripts/pipeline/check_ssh_recipe.py \
        --run-root results/runs/sacm_gcd_ssh_full_ab \
        --config configs/ablations/sacm_gcd_ssh_full_ab.yaml \
        --arm ssh --seeds 0 1 2 3 --epochs 800
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REQUIRED_KEYS = (
    'specialized_heads',
    'specialized_head_weight',
    'head_fusion',
    'fusion_loss_weight',
    'head_fusion_mode',
    'head_fusion_warmup_epochs',
    'specialized_topology_mode',
    'head_select',
    'deployment_head',
)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None


def _expected_recipe(config_path: Path, arm: str) -> dict:
    try:
        with open(config_path) as handle:
            config = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f'cannot read {config_path}: {exc}')
    common = config.get('common_train_args') or {}
    arms = config.get('arms') or {}
    arm_spec = arms.get(arm) or {}
    if not isinstance(common, dict) or not isinstance(arm_spec, dict):
        raise SystemExit(f'{config_path}: invalid common_train_args/arms')
    merged = {}
    for mapping in (common, arm_spec.get('args') or {}):
        if not isinstance(mapping, dict):
            raise SystemExit(f'{config_path}: invalid {arm} args')
        for key, value in mapping.items():
            merged[str(key).replace('-', '_')] = value
    missing = [key for key in REQUIRED_KEYS if key not in merged]
    if missing:
        raise SystemExit(
            f'{config_path}: arm {arm!r} is missing recipe keys {missing}')
    return merged


def _recipe_mismatches(adapter: dict, expected: dict) -> list[str]:
    mismatches = []
    for key, want in expected.items():
        if key not in adapter:
            continue
        got = adapter[key]
        numeric = (
            isinstance(want, (int, float))
            and not isinstance(want, bool)
            and isinstance(got, (int, float))
            and not isinstance(got, bool)
        )
        if numeric:
            if abs(float(got) - float(want)) > 1e-6:
                mismatches.append(f'{key}: {got!r} != {want!r}')
        elif str(got) != str(want):
            mismatches.append(f'{key}: {got!r} != {want!r}')
    return mismatches


def _check_checkpoint(path: Path, expected: dict) -> list[str]:
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        return [f'cannot load {path}: {exc}']
    except Exception as exc:
        return [f'cannot load {path}: {exc}']
    if not isinstance(payload, dict):
        return [f'{path}: checkpoint is not a mapping']
    state = payload.get('model_state_dict')
    if not isinstance(state, dict) or not state:
        return [f'{path}: model_state_dict is missing or empty']
    config = payload.get('config')
    adapter = config.get('adapter_config') if isinstance(config, dict) else None
    if not isinstance(adapter, dict):
        return [f'{path}: config.adapter_config is missing']
    missing = [key for key in REQUIRED_KEYS if key not in adapter]
    if missing:
        return [f'{path}: adapter config is missing {missing}']
    return _recipe_mismatches(adapter, expected)


def _run_epochs(run_root: Path):
    info = _read_json(run_root / 'run_info.json')
    if isinstance(info, dict):
        try:
            return info.get('status'), int(info.get('epochs'))
        except (TypeError, ValueError):
            return info.get('status'), None
    return None, None


def _resume_epochs(run_root: Path):
    config = _read_json(run_root / 'train_config.json')
    if not isinstance(config, dict):
        return None
    args = config.get('args')
    if not isinstance(args, dict):
        return None
    try:
        return int(args.get('epochs'))
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--arm', default='ssh')
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args(argv)

    expected = _expected_recipe(args.config, args.arm)
    checked = 0
    errors = []
    for seed in sorted(set(args.seeds)):
        seed_root = args.run_root / 'ssh' / f'seed_{seed}'
        status, recorded_epochs = _run_epochs(seed_root)
        complete = (
            status == 'done'
            and recorded_epochs is not None
            and recorded_epochs == args.epochs
        )
        if complete:
            checkpoint = seed_root / 'best_model.pth'
            if not checkpoint.is_file():
                errors.append(
                    f'seed {seed}: complete run has no {checkpoint.name}')
                continue
            checked += 1
            mismatches = _check_checkpoint(checkpoint, expected)
            if mismatches:
                errors.append(
                    f'seed {seed}: {checkpoint.name} recipe mismatch: '
                    + '; '.join(mismatches))
            else:
                print(
                    f'seed {seed}: existing {args.epochs}-epoch SSH '
                    'checkpoint recipe matches')
            continue

        resume_path = seed_root / '.resume_state.pth'
        if resume_path.is_file() and _resume_epochs(
                seed_root) == args.epochs:
            checked += 1
            mismatches = _check_checkpoint(resume_path, expected)
            if mismatches:
                errors.append(
                    f'seed {seed}: {resume_path.name} recipe mismatch: '
                    + '; '.join(mismatches))
            else:
                print(
                    f'seed {seed}: resumable {args.epochs}-epoch SSH '
                    'checkpoint recipe matches')
        else:
            print(f'seed {seed}: no matching completed/resumed run; '
                  'training will start fresh')

    if errors:
        for item in errors:
            print(f'ERROR: {item}', file=sys.stderr)
        print(
            'Archive/move the affected seed run roots, or rerun with '
            'RESUME=0 to retrain from the YAML recipe.', file=sys.stderr)
        return 1
    print(f'SSH recipe check OK ({checked} existing checkpoint(s))')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
