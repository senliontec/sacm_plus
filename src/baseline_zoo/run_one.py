"""Run one self-contained baseline adapter.

    python -m baseline_zoo.run_one --model unet --stage train ...
    python -m baseline_zoo.run_one --model unet --stage predict ...

This is the only per-model entrypoint used by configs and shell wrappers.
It never imports from the disposable ``baselines/`` reference tree.
"""

from __future__ import annotations

import argparse

from .adapters.registry import available_adapters, load_adapter
from .runtime import main_for


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='baseline_zoo.run_one', add_help=False,
        description=f'available adapters: {", ".join(available_adapters())}')
    parser.add_argument('--model', required=True)
    known, rest = parser.parse_known_args(argv)
    try:
        adapter = load_adapter(known.model)
    except KeyError as exc:
        parser.error(str(exc.args[0]))
    return main_for(adapter, rest)


if __name__ == '__main__':
    raise SystemExit(main())
