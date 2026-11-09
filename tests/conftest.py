"""Pytest configuration: repo-root import path + slow-test marker."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run heavy tests (persistence engines); skipped by default",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-slow'):
        return
    skip = pytest.mark.skip(reason='slow test; pass --run-slow to run it')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip)
