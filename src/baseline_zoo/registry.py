"""Model registry backed by ``configs/models/*.yaml``.

The registry is deliberately metadata-only.  A model becomes available when
its YAML declares ``status: available`` and a train/eval entrypoint; the
runner can then plan commands without importing model-specific packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / 'configs' / 'models'
PROTOCOLS_DIR = REPO_ROOT / 'configs' / 'protocols'
VALID_STATUS = {'available', 'reference_only', 'unavailable'}


@dataclass(frozen=True)
class ModelConfig:
    id: str
    display_name: str
    family: str
    status: str
    train_entrypoint: str | None
    eval_entrypoint: str | None
    checkpoint_template: str | None
    note: str | None
    protocols: tuple[str, ...]
    defaults: dict[str, Any]
    path: Path

    @property
    def available(self) -> bool:
        return self.status == 'available'


def _load_one(path: Path) -> ModelConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f'{path}: model config must be a mapping')

    model_id = raw.get('id')
    if not model_id:
        raise ValueError(f'{path}: missing required field "id"')
    status = raw.get('status', 'available')
    if status not in VALID_STATUS:
        raise ValueError(
            f'{path}: status must be one of {sorted(VALID_STATUS)}, got {status!r}')

    train = raw.get('train_entrypoint')
    eval_entrypoint = raw.get('eval_entrypoint')
    if status == 'available' and (not train or not eval_entrypoint):
        raise ValueError(
            f'{path}: available model needs train_entrypoint and eval_entrypoint')

    return ModelConfig(
        id=str(model_id),
        display_name=str(raw.get('display_name', model_id)),
        family=str(raw.get('family', 'unknown')),
        status=str(status),
        train_entrypoint=None if train is None else str(train),
        eval_entrypoint=None if eval_entrypoint is None else str(eval_entrypoint),
        checkpoint_template=None if raw.get('checkpoint') is None
        else str(raw['checkpoint']),
        note=None if raw.get('note') is None else str(raw['note']),
        protocols=tuple(raw.get('protocols', ())),
        defaults=dict(raw.get('defaults', {})),
        path=path,
    )


def load_models(directory: Path = MODELS_DIR) -> dict[str, ModelConfig]:
    """Load every ``*.yaml`` model config keyed by model id."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'model config directory not found: {directory}')

    models: dict[str, ModelConfig] = {}
    for path in sorted(directory.glob('*.yaml')):
        model = _load_one(path)
        if model.id in models:
            raise ValueError(
                f'duplicate model id {model.id!r}: '
                f'{models[model.id].path} and {path}')
        models[model.id] = model
    return models


def resolve_models(requested: str, models: dict[str, ModelConfig]) -> list[ModelConfig]:
    """Resolve ``all`` or a comma-separated list to ModelConfig objects."""
    available = {key: value for key, value in models.items() if value.available}
    if requested.strip() == 'all':
        return [available[key] for key in sorted(available)]

    names = [name.strip() for name in requested.split(',') if name.strip()]
    if not names:
        raise ValueError('--model cannot be empty')
    # Duplicate names would schedule colliding jobs for the same output
    # root; keep the first occurrence only.
    names = list(dict.fromkeys(names))
    unknown = [name for name in names if name not in models]
    if unknown:
        raise ValueError(
            f'unknown model(s) {unknown}; available: {sorted(models)}')
    unavailable = [name for name in names if not models[name].available]
    if unavailable:
        details = ', '.join(
            f'{name}={models[name].status}' for name in unavailable)
        raise ValueError(f'model(s) not available: {details}')
    return [models[name] for name in names]
