"""Training/evaluation provenance and resumable-artifact helpers.

The goal is that one run directory is self-describing:

* full resolved training arguments (not just the subset used by the model);
* code revision, dirty state, and environment versions;
* data roots, protocol/split manifest copies and SHA256 hashes;
* base checkpoint path/hash;
* RNG and DataLoader-generator state for resumable checkpoints;
* W&B run identity;
* a prediction index that maps every saved prediction to its GT mask.

Everything is written atomically so a crash cannot leave a half-written
JSON file.  The module deliberately has no project imports beyond numpy and
torch, avoiding cycles with ``core.trainer`` / ``core.evaluator``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ARTIFACT_VERSION = 2
_SECRET_MARKERS = ('api_key', 'token', 'secret', 'password', 'credential')
_MANIFEST_NAMES = (
    'protocol_manifest.csv',
    'split_manifest.csv',
    'protocol_summary.json',
)
_IMAGE_SUFFIXES = (
    '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm', '.pgm',
)


# ---------------------------------------------------------------------------
# JSON / hashing helpers
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    """Convert argparse/Path/numpy values to JSON-safe primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return str(value)


def redact_args(args: Any) -> dict[str, Any]:
    """Return a full, log-safe snapshot of an argparse/SimpleNamespace."""
    values = vars(args) if hasattr(args, '__dict__') else dict(args)
    safe: dict[str, Any] = {}
    for key, value in values.items():
        if any(marker in str(key).lower() for marker in _SECRET_MARKERS):
            safe[key] = '***' if value not in (None, '') else value
        else:
            safe[key] = _json_safe(value)
    return safe


def write_json_atomic(path: Path | str, payload: Any) -> None:
    """Write JSON atomically; callers decide whether an OSError is fatal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(tmp, path)


def read_json(path: Path | str) -> dict:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# ---------------------------------------------------------------------------
# Environment / git / checkpoint provenance
# ---------------------------------------------------------------------------

def git_info(repo_root: Path | str | None = None) -> dict[str, Any]:
    root = Path(repo_root or Path(__file__).resolve().parents[2])

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ['git', '-C', str(root), *args],
                check=False, capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run('rev-parse', 'HEAD')
    branch = run('rev-parse', '--abbrev-ref', 'HEAD')
    status = run('status', '--porcelain')
    diff = run('diff', '--no-ext-diff')
    return {
        'repo_root': str(root),
        'commit': commit,
        'branch': branch,
        'dirty': bool(status),
        'porcelain': status,
        'diff_sha256': sha256_text(diff) if diff else None,
    }


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, '__version__', None)


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        'python': sys.version,
        'executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'hostname': socket.gethostname(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': (torch.backends.cudnn.version()
                  if torch.backends.cudnn.is_available() else None),
        'numpy': np.__version__,
        'scipy': _module_version('scipy'),
        'skimage': _module_version('skimage'),
        'PIL': _module_version('PIL'),
        'gpu_count': 0,
        'gpus': [],
    }
    try:
        if torch.cuda.is_available():
            info['gpu_count'] = int(torch.cuda.device_count())
            info['gpus'] = [
                torch.cuda.get_device_name(index)
                for index in range(info['gpu_count'])
            ]
    except Exception:
        pass
    return info


def base_checkpoint_info(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {'path': None, 'sha256': None, 'size': None, 'error': None}
    candidate = Path(path)
    payload = {
        'path': str(candidate),
        'resolved': str(candidate.resolve()) if candidate.exists() else None,
        'sha256': None,
        'size': None,
        'error': None,
    }
    if not candidate.is_file():
        payload['error'] = 'not a file'
        return payload
    try:
        payload['size'] = int(candidate.stat().st_size)
        payload['sha256'] = sha256_file(candidate)
    except OSError as exc:
        payload['error'] = f'{type(exc).__name__}: {exc}'
    return payload


# ---------------------------------------------------------------------------
# Data manifests
# ---------------------------------------------------------------------------

def _resolve_protocol(data_root: Path,
                      protocol: Path | str | None) -> Path | None:
    if protocol:
        candidate = Path(protocol)
        if candidate.is_file():
            return candidate
    candidate = data_root / 'protocol_manifest.csv'
    return candidate if candidate.is_file() else None


def _resolve_data_path(raw: str, data_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    for base in (Path.cwd(), repo_root, data_root, data_root.parent):
        candidate = base / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _file_info(path: Path, *, base: Path | None = None,
               hash_file: bool = False) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        return {
            'path': str(path),
            'relative_path': None,
            'size': None,
            'mtime_ns': None,
            'sha256': None,
            'error': f'{type(exc).__name__}: {exc}',
        }
    relative = None
    if base is not None:
        try:
            relative = str(path.relative_to(base))
        except ValueError:
            relative = None
    payload = {
        'path': str(path),
        'relative_path': relative,
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
        'sha256': None,
        'error': None,
    }
    if hash_file:
        try:
            payload['sha256'] = sha256_file(path)
        except OSError as exc:
            payload['error'] = f'{type(exc).__name__}: {exc}'
    return payload


def _index_directory(directory: Path, *, base: Path,
                     hash_files: bool) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    return [
        _file_info(path, base=base, hash_file=hash_files)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]


def build_data_manifest(
    data_root: Path | str,
    *,
    protocol: Path | str | None = None,
    hash_files: bool = True,
) -> dict[str, Any]:
    """Describe train/val/test data without copying the images themselves."""
    root = Path(data_root)
    protocol_path = _resolve_protocol(root, protocol)
    manifest: dict[str, Any] = {
        'artifact_version': ARTIFACT_VERSION,
        'created_at': _utc_now(),
        'data_root': str(root),
        'hash_files': bool(hash_files),
        'splits': {},
        'manifests': {},
        'test_datasets': {},
    }
    for split in ('train', 'val'):
        manifest['splits'][split] = {
            'images': _index_directory(
                root / split / 'images', base=root, hash_files=hash_files),
            'masks': _index_directory(
                root / split / 'masks', base=root, hash_files=hash_files),
        }
    for name in _MANIFEST_NAMES:
        path = root / name
        if path.is_file():
            manifest['manifests'][name] = _file_info(
                path, base=root, hash_file=True)
    if protocol_path is not None:
        manifest['protocol'] = _file_info(
            protocol_path, base=root, hash_file=True)
        rows_by_dataset: dict[str, list[Path]] = {}
        with open(protocol_path, newline='') as handle:
            for row in csv.DictReader(handle):
                if not str(row.get('split', '')).endswith('_test'):
                    continue
                name = str(row.get('dataset', '')).strip()
                mask_raw = str(row.get('mask_path', '')).strip()
                if not name or not mask_raw:
                    continue
                rows_by_dataset.setdefault(name, []).append(
                    _resolve_data_path(mask_raw, root))
        for name, masks in sorted(rows_by_dataset.items()):
            unique = sorted(set(masks))
            root_dir = unique[0].parent.parent if unique else None
            mask_dir = unique[0].parent if unique else None
            manifest['test_datasets'][name] = {
                'root': str(root_dir) if root_dir else None,
                'mask_dir': str(mask_dir) if mask_dir else None,
                'n_masks': len(unique),
                'masks': [
                    _file_info(path, base=root_dir, hash_file=hash_files)
                    for path in unique
                ],
            }
    return manifest


def _fingerprint_value(value: Any) -> Any:
    """Drop volatile metadata (creation time, mtimes) from the fingerprint."""
    if isinstance(value, dict):
        return {
            key: _fingerprint_value(item)
            for key, item in value.items()
            if key not in ('created_at', 'mtime_ns')
        }
    if isinstance(value, list):
        return [_fingerprint_value(item) for item in value]
    return value


def data_manifest_fingerprint(manifest: dict[str, Any]) -> str:
    """Stable SHA256 of a data manifest.

    Contents are identified by size/relative path/SHA256; creation time and
    filesystem mtimes are deliberately excluded so a copied dataset with the
    same bytes can still resume.
    """
    stable = _fingerprint_value(manifest)
    return sha256_text(json.dumps(
        _json_safe(stable), sort_keys=True, separators=(',', ':')))


def _copy_manifest_files(data_root: Path, provenance_dir: Path,
                         protocol: Path | None) -> dict[str, str]:
    copied: dict[str, str] = {}
    candidates = []
    if protocol is not None:
        candidates.append(protocol)
    candidates += [data_root / name for name in _MANIFEST_NAMES]
    seen: set[Path] = set()
    for source in candidates:
        if source in seen or not source.is_file():
            continue
        seen.add(source)
        target = provenance_dir / source.name
        try:
            shutil.copy2(source, target)
            copied[source.name] = str(target)
        except OSError:
            pass
    return copied


def prepare_training_artifacts(
    args: Any,
    save_dir: Path | str,
    *,
    repo_root: Path | str | None = None,
    hash_files: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Write train_config.json + provenance/ and return their paths."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir = save_dir / 'provenance'
    provenance_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(getattr(args, 'data_root', '.'))
    protocol = _resolve_protocol(
        data_root, getattr(args, 'protocol', None))
    data_manifest = build_data_manifest(
        data_root, protocol=protocol, hash_files=hash_files)
    data_manifest_path = provenance_dir / 'data_manifest.json'
    write_json_atomic(data_manifest_path, data_manifest)
    data_fingerprint = data_manifest_fingerprint(data_manifest)
    copied = _copy_manifest_files(data_root, provenance_dir, protocol)

    environment = environment_info()
    git = git_info(repo_root)
    base = base_checkpoint_info(getattr(args, 'checkpoint', None))
    config_payload = {
        'artifact_version': ARTIFACT_VERSION,
        'created_at': _utc_now(),
        'resume': bool(resume),
        'args': redact_args(args),
        'git': git,
        'environment': environment,
        'base_checkpoint': base,
        'data_manifest': {
            'path': str(data_manifest_path),
            'sha256': data_fingerprint,
        },
        'copied_manifests': copied,
    }
    write_json_atomic(provenance_dir / 'environment.json', environment)
    write_json_atomic(provenance_dir / 'git.json', git)
    config_name = 'train_config.json'
    if resume and (save_dir / config_name).is_file():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        config_name = f'train_config.resume_{stamp}.json'
    config_path = save_dir / config_name
    write_json_atomic(config_path, config_payload)

    return {
        'artifact_version': ARTIFACT_VERSION,
        'save_dir': str(save_dir),
        'provenance_dir': str(provenance_dir),
        'train_config': str(config_path),
        'train_config_payload': config_payload,
        'data_manifest': str(data_manifest_path),
        'data_manifest_payload': data_manifest,
        'data_manifest_sha256': data_fingerprint,
        'copied_manifests': copied,
        'git': git,
        'environment': environment,
        'base_checkpoint': base,
    }


# ---------------------------------------------------------------------------
# W&B and reproducibility state
# ---------------------------------------------------------------------------

def wandb_info(run: Any) -> dict[str, Any]:
    if run is None:
        return {}
    info = {}
    for key in ('id', 'name', 'project', 'entity', 'url', 'mode', 'dir'):
        try:
            value = getattr(run, key, None)
        except Exception:
            value = None
        if value is not None:
            info[key] = str(value)
    return info


def write_wandb_info(save_dir: Path | str, run: Any) -> dict[str, Any]:
    info = wandb_info(run)
    if info:
        write_json_atomic(Path(save_dir) / 'wandb_run.json', info)
    return info


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': [],
    }
    try:
        if torch.cuda.is_available():
            state['cuda'] = [
                torch.cuda.get_rng_state(index).cpu()
                for index in range(torch.cuda.device_count())
            ]
    except Exception:
        state['cuda'] = []
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    try:
        if 'python' in state:
            random.setstate(tuple(state['python']))
    except Exception:
        pass
    try:
        if 'numpy' in state:
            np.random.set_state(state['numpy'])
    except Exception:
        pass
    try:
        if 'torch' in state:
            torch.set_rng_state(state['torch'])
    except Exception:
        pass
    try:
        cuda_states = state.get('cuda') or []
        if torch.cuda.is_available() and cuda_states:
            count = min(len(cuda_states), torch.cuda.device_count())
            for index in range(count):
                torch.cuda.set_rng_state(cuda_states[index], device=index)
    except Exception:
        pass


def capture_generator_state(generator) -> Any:
    if generator is None:
        return None
    try:
        return generator.get_state()
    except Exception:
        return None


def restore_generator_state(generator, state) -> None:
    if generator is None or state is None:
        return
    try:
        tensor = state if isinstance(state, torch.Tensor) else torch.as_tensor(state)
        generator.set_state(tensor.cpu())
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Prediction index shared by evaluate.py and baseline runtime
# ---------------------------------------------------------------------------

PREDICTION_INDEX_FIELDS = (
    'dataset', 'image', 'pred_path', 'gt_path', 'image_path',
)


def write_prediction_index(
    out_dir: Path | str,
    dataset: str,
    rows: Sequence[dict[str, Any]],
    *,
    manifest_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write prediction_index.csv and eval_manifest.json for later audits."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / 'prediction_index.csv'
    tmp = index_path.with_name(index_path.name + '.tmp')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(PREDICTION_INDEX_FIELDS),
            extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, index_path)
    manifest = {
        'artifact_version': ARTIFACT_VERSION,
        'created_at': _utc_now(),
        'dataset': str(dataset),
        'output_dir': str(out_dir.resolve()),
        'prediction_index': str(index_path.resolve()),
        'prediction_suffix': '_result.png',
        'n_images': len(rows),
    }
    if manifest_extra:
        manifest.update(_json_safe(manifest_extra))
    manifest_path = out_dir / 'eval_manifest.json'
    write_json_atomic(manifest_path, manifest)
    return manifest
