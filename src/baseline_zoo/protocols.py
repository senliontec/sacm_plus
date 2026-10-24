"""Seed/protocol expansion backed by ``configs/protocols/*.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .registry import PROTOCOLS_DIR, REPO_ROOT


@dataclass(frozen=True)
class SeedSplit:
    seed: int
    split_root: Path
    train_manifest: Path
    test_manifest: Path


@dataclass(frozen=True)
class ProtocolConfig:
    name: str
    description: str
    seeds: tuple[int, ...]
    split_root_template: str
    train_manifest_template: str
    test_manifest_template: str
    path: Path

    def select_seeds(self, requested: list[int] | None = None) -> list[int]:
        if not requested:
            return list(self.seeds)
        unknown = sorted(set(requested) - set(self.seeds))
        if unknown:
            raise ValueError(
                f'protocol {self.name!r} has seeds {list(self.seeds)}, '
                f'requested unknown {unknown}')
        return list(dict.fromkeys(requested))

    def split_for(self, seed: int) -> SeedSplit:
        if seed not in self.seeds:
            raise ValueError(
                f'protocol {self.name!r} has seeds {list(self.seeds)}, '
                f'requested {seed}')
        split_root = REPO_ROOT / self.split_root_template.format(seed=seed)
        train_manifest = REPO_ROOT / self.train_manifest_template.format(seed=seed)
        test_manifest = REPO_ROOT / self.test_manifest_template.format(seed=seed)
        return SeedSplit(
            seed=seed,
            split_root=split_root,
            train_manifest=train_manifest,
            test_manifest=test_manifest,
        )


def _load_one(path: Path) -> ProtocolConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f'{path}: protocol config must be a mapping')

    required = ('name', 'seeds', 'split_root', 'train_manifest',
                'test_manifest')
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f'{path}: missing required fields {missing}')

    seeds = tuple(int(seed) for seed in raw['seeds'])
    if not seeds:
        raise ValueError(f'{path}: seeds cannot be empty')
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'{path}: duplicate seeds {seeds}')

    return ProtocolConfig(
        name=str(raw['name']),
        description=str(raw.get('description', '')),
        seeds=seeds,
        split_root_template=str(raw['split_root']),
        train_manifest_template=str(raw['train_manifest']),
        test_manifest_template=str(raw['test_manifest']),
        path=path,
    )


def load_protocol(name: str, directory: Path = PROTOCOLS_DIR) -> ProtocolConfig:
    path = Path(directory) / f'{name}.yaml'
    if not path.is_file():
        raise FileNotFoundError(f'protocol config not found: {path}')
    return _load_one(path)


def load_protocols(directory: Path = PROTOCOLS_DIR) -> dict[str, ProtocolConfig]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'protocol config directory not found: {directory}')
    protocols: dict[str, ProtocolConfig] = {}
    for path in sorted(directory.glob('*.yaml')):
        protocol = _load_one(path)
        if protocol.name in protocols:
            raise ValueError(f'duplicate protocol name {protocol.name!r}')
        protocols[protocol.name] = protocol
    return protocols
