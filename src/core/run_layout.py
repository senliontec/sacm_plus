"""Canonical results layout shared by baselines and SACM-X.

Every completed run has exactly one user-facing directory::

    results/runs/<method>/<variant>/seed_<seed>/

``method`` is the baseline id or ``sacm_x``.  ``variant`` is ``default`` for
baselines and the SACM-X preset (for example ``full_typed``).  The seed is
always part of the path so two seeds can never overwrite each other.

The layout is intentionally free of pipeline-specific details.  Training and
evaluation code should derive every output path from these helpers instead of
hard-coding ``results/paper`` or ``results/sacm_x``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / 'results' / 'runs'
DEFAULT_VARIANT = 'default'


def normalize_variant(variant: str | None) -> str:
    """Return a filesystem-safe variant name (never empty)."""
    value = str(variant or '').strip()
    return value or DEFAULT_VARIANT


def default_variant(defaults: dict | None) -> str:
    """Baseline configs use ``default``; SACM-X uses its configured preset."""
    return normalize_variant((defaults or {}).get('preset'))


def run_root(method: str, variant: str | None, seed: int,
             root: Path | str | None = None) -> Path:
    """Return ``<root>/<method>/<variant>/seed_<seed>``."""
    if not method:
        raise ValueError('method must be a non-empty string')
    base = Path(root) if root is not None else RUNS_ROOT
    return base / str(method) / normalize_variant(variant) / f'seed_{int(seed)}'


def eval_root(method: str, variant: str | None, seed: int,
              root: Path | str | None = None) -> Path:
    return run_root(method, variant, seed, root) / 'eval'


def canonical_paths(method: str, variant: str | None, seed: int,
                    root: Path | str | None = None) -> dict[str, Path]:
    """Return the paths every run must expose."""
    run = run_root(method, variant, seed, root)
    evaluation = run / 'eval'
    return {
        'run_root': run,
        'checkpoint': run / 'best_model.pth',
        'run_info': run / 'run_info.json',
        'metrics_history': run / 'metrics_history.json',
        'train_log': run / 'train.log',
        'predictions': run / 'predictions',
        'eval_root': evaluation,
        'eval_log': evaluation / 'eval.log',
        'jobs': evaluation / 'jobs.csv',
        'manifest': evaluation / 'run_manifest.json',
        'tables': evaluation / 'tables',
        'all_metrics': evaluation / 'tables' / 'all_metrics.csv',
        'macro_metrics': evaluation / 'tables' / 'macro_metrics.csv',
        'final_report': evaluation / 'tables' / 'final_report.md',
    }


__all__ = [
    'DEFAULT_VARIANT', 'REPO_ROOT', 'RUNS_ROOT', 'canonical_paths',
    'default_variant', 'eval_root', 'normalize_variant', 'run_root',
]
