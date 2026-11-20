"""Local runner for CoSNA.

    PYTHONPATH=src python -m methods.cosna.run_one --stage train ...
    PYTHONPATH=src python -m methods.cosna.run_one --stage predict ...
"""

from __future__ import annotations

from baseline_zoo.runtime import main_for

from .adapter import CoSNAAdapter


def main(argv: list[str] | None = None) -> int:
    return main_for(CoSNAAdapter(), argv)


if __name__ == '__main__':
    raise SystemExit(main())
