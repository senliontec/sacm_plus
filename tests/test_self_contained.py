"""The runtime must survive deletion of the ``baselines/`` reference tree."""

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / 'src'


def _is_reference_module(name: str) -> bool:
    return name == 'baselines' or name.startswith('baselines.')


def test_src_does_not_import_reference_tree():
    violations = []
    for path in SRC.rglob('*.py'):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_reference_module(alias.name):
                        violations.append(f'{path}: import {alias.name}')
            elif isinstance(node, ast.ImportFrom):
                if node.module and _is_reference_module(node.module):
                    violations.append(f'{path}: from {node.module}')
            elif isinstance(node, ast.Call):
                func = node.func
                is_path_call = (
                    isinstance(func, ast.Attribute)
                    and func.attr in {'insert', 'append'}
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == 'path'
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == 'sys')
                if is_path_call:
                    for arg in node.args:
                        if (isinstance(arg, ast.Constant)
                                and isinstance(arg.value, str)
                                and 'baselines' in arg.value):
                            violations.append(f'{path}: sys.path <- {arg.value}')
    assert not violations, '\n'.join(violations)


def test_unet_imports_from_self_contained_package(tmp_path):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC)
    code = (
        'import baseline_zoo.adapters.unet as m;'
        'print(m.UNet.__module__)')
    output = subprocess.check_output(
        [sys.executable, '-c', code], cwd=tmp_path, env=env, text=True)
    assert output.strip() == 'baseline_zoo.models.unet.unet_model'


def test_run_one_resolves_unet(tmp_path):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(SRC)
    output = subprocess.check_output(
        [sys.executable, '-m', 'baseline_zoo.run_one', '--model', 'unet',
         '--help'], cwd=tmp_path, env=env, text=True, stderr=subprocess.STDOUT)
    assert '--stage' in output


def test_all_registered_adapters_import():
    sys.path.insert(0, str(SRC))
    from baseline_zoo.adapters.registry import (
        available_adapters,
        load_adapter,
    )
    for model_id in available_adapters():
        adapter = load_adapter(model_id)
        assert adapter.id == model_id
