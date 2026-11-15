"""Local runner for TopoStrand-SAM.

This entrypoint imports the new adapter directly, so the model never needs a
registry entry inside ``baseline_zoo.adapters``.  It still uses the shared
runtime for the 18/6/20 protocol, metrics, logs and checkpoint schema.

    PYTHONPATH=src python -m methods.topostrand.run_one --stage train ...
"""

from __future__ import annotations

from baseline_zoo.runtime import main_for

from .adapter import TopoStrandAdapter


def main(argv: list[str] | None = None) -> int:
    return main_for(TopoStrandAdapter(), argv)


if __name__ == '__main__':
    raise SystemExit(main())
