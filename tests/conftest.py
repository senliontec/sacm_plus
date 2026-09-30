"""Pytest configuration: repo-root import path + slow-test marker."""

import sys
from pathlib import Path

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
