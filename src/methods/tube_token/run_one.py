"""Entry point for TubeToken-SAM.

The shared runtime owns the epoch loop, validation, checkpointing, W&B and
prediction contract.  We only supply the new adapter and the 800-epoch
default (DCD's canonical budget) without adding a baseline registry entry.
"""

from __future__ import annotations

import sys

import baseline_zoo.runtime as runtime

from .adapter import TubeTokenAdapter


def main(argv: list[str] | None = None) -> int:
    adapter = TubeTokenAdapter()
    original = runtime._configured_epochs

    def _configured_epochs(_adapter):
        return 800

    runtime._configured_epochs = _configured_epochs
    try:
        return runtime.main_for(
            adapter, sys.argv[1:] if argv is None else argv)
    finally:
        runtime._configured_epochs = original


if __name__ == '__main__':
    raise SystemExit(main())
