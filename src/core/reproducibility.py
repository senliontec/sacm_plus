"""Deterministic execution helpers.

This module centralizes the settings that make two runs on the same
hardware/software stack reproducible:

* Python/NumPy/PyTorch/CUDA seeds;
* deterministic cuDNN/cuBLAS settings and TF32 disabled;
* PyTorch deterministic algorithms (strict by default);
* deterministic DataLoader worker seeding;
* a small environment manifest for checkpoints/W&B.

Bitwise reproducibility across different GPUs, drivers, CUDA/PyTorch
versions, or worker counts is not guaranteed.  With ``strict=True`` an
operation without a deterministic CUDA implementation raises instead of
silently producing a non-reproducible result; use
``--deterministic_strict false`` only for a best-effort run.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_determinism(
    seed: int,
    *,
    deterministic: bool = True,
    strict: bool = True,
    num_threads: int = 1,
    cuda_seed_all: bool = True,
) -> dict:
    """Seed all RNGs and configure deterministic kernels.

    ``cuda_seed_all=False`` is used before DDP initialization so we do not
    initialize every visible GPU in every rank; the caller seeds the selected
    local device after ``torch.cuda.set_device``.
    """
    # cuBLAS deterministic workspace must be set before the first CUDA
    # matmul/CUDA context use.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_seed_all and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    info = {
        'seed': seed,
        'deterministic': bool(deterministic),
        'strict': bool(strict),
        'num_threads': int(num_threads),
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        'cublas_workspace_config': os.environ.get(
            'CUBLAS_WORKSPACE_CONFIG'),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available() else None,
        'device': None,
    }
    # Avoid touching a CUDA device before DDP has selected its local rank.
    if cuda_seed_all and torch.cuda.is_available():
        try:
            info['device'] = torch.cuda.get_device_name(
                torch.cuda.current_device())
        except Exception:
            info['device'] = None

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TF32 reductions can differ between runs; disable both paths.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if num_threads and int(num_threads) > 0:
            torch.set_num_threads(int(num_threads))
        # strict=True raises when an op has no deterministic CUDA kernel.
        torch.use_deterministic_algorithms(
            True, warn_only=not bool(strict))
    return info


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy inside each DataLoader worker.

    PyTorch already seeds ``torch`` per worker via ``base_seed +
    worker_id``; Python's ``random`` and NumPy are not necessarily covered
    by older loader paths, so seed them explicitly.
    """
    worker_seed = int(torch.initial_seed()) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
