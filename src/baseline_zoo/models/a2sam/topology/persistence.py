"""Exact 2D superlevel-set persistence for A2SAM.

The local DMT/dipha pipeline is not usable offline (it needs an OpenMPI-4
runtime).  We therefore compute exact 0D persistence with the installed
``cripser`` backend and reconstruct the Morse/branch geometry separately in
``morse.py``.  This is the documented compromise: persistence values are
exact; the discrete gradient field and stable manifolds are reconstructed.

cripser is used as an installed dependency (LGPL-3.0), not vendored.  The
``gudhi`` backend is an explicit MIT fallback and is never selected silently.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PersistenceBars:
    """H0 bars in probability units.

    ``birth_rc``/``death_rc`` use NumPy ``(row, col)`` order.  Essential bars
    have ``death_rc=(-1,-1)`` and ``essential=True``.
    """

    birth_prob: np.ndarray
    death_prob: np.ndarray
    persistence: np.ndarray
    birth_rc: np.ndarray
    death_rc: np.ndarray
    essential: np.ndarray

    def __len__(self) -> int:
        return int(self.birth_prob.shape[0])

    @property
    def finite(self) -> np.ndarray:
        return ~self.essential

    def sorted(self) -> 'PersistenceBars':
        order = np.argsort(-self.persistence, kind='stable')
        return PersistenceBars(
            birth_prob=self.birth_prob[order],
            death_prob=self.death_prob[order],
            persistence=self.persistence[order],
            birth_rc=self.birth_rc[order],
            death_rc=self.death_rc[order],
            essential=self.essential[order],
        )


def _validate_probability(prob: np.ndarray) -> np.ndarray:
    array = np.asarray(prob, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f'probability map must be 2-D, got {array.shape}')
    if not np.isfinite(array).all():
        raise ValueError('probability map contains NaN/inf')
    return array


def resolve_backend(backend: str) -> str:
    """Resolve ``auto`` to an installed exact persistence backend."""
    if backend != 'auto':
        return backend
    if importlib.util.find_spec('cripser'):
        return 'cripser'
    if importlib.util.find_spec('gudhi'):
        return 'gudhi'
    raise RuntimeError(
        'A2SAM topology requires cripser or gudhi; install an offline '
        'wheel or select a different backend')


def compute_persistence(
    prob: np.ndarray,
    *,
    backend: str = 'cripser',
    min_persistence: float = 0.0,
) -> PersistenceBars:
    """Return exact H0 superlevel-set bars for a 2D likelihood map."""
    array = _validate_probability(prob)
    backend = resolve_backend(backend)
    if backend == 'cripser':
        bars = _compute_cripser(array)
    elif backend == 'gudhi':
        bars = _compute_gudhi(array)
    elif backend == 'join_tree':
        bars = _compute_join_tree(array)
    else:
        raise ValueError(f'unknown topology backend: {backend!r}')
    if min_persistence > 0:
        keep = bars.persistence >= float(min_persistence)
        bars = PersistenceBars(
            birth_prob=bars.birth_prob[keep],
            death_prob=bars.death_prob[keep],
            persistence=bars.persistence[keep],
            birth_rc=bars.birth_rc[keep],
            death_rc=bars.death_rc[keep],
            essential=bars.essential[keep],
        )
    return bars


def _compute_cripser(prob: np.ndarray) -> PersistenceBars:
    try:
        import cripser
    except ImportError as exc:  # pragma: no cover - doctor catches this
        raise RuntimeError(
            'A2SAM topology requires cripser; install the topology extra '
            'or select --a2sam-topology-backend gudhi') from exc
    # cripser computes sublevel persistence.  Negating the likelihood gives
    # superlevel persistence.  Columns are
    # [dim,birth,death,brow,bcol,bz,drow,dcol,dz] in this build (verified
    # against asymmetric arrays).
    table = cripser.compute_ph(
        -prob.astype(np.float64, copy=False),
        maxdim=0,
        n_threads=1,
        location='yes',
    )
    table = np.asarray(table)
    if table.size == 0:
        return PersistenceBars(
            birth_prob=np.zeros(0, np.float32),
            death_prob=np.zeros(0, np.float32),
            persistence=np.zeros(0, np.float32),
            birth_rc=np.zeros((0, 2), np.int32),
            death_rc=np.zeros((0, 2), np.int32),
            essential=np.zeros(0, bool),
        )
    table = table[table[:, 0] == 0]
    finite = np.isfinite(table[:, 2]) & (table[:, 2] < np.finfo(np.float64).max)
    birth_prob = (-table[:, 1]).astype(np.float32)
    death_prob = np.where(finite, -table[:, 2], 0.0).astype(np.float32)
    persistence = np.where(finite, birth_prob - death_prob,
                           birth_prob).astype(np.float32)
    birth_rc = table[:, 3:5].astype(np.int32)
    death_rc = np.where(
        finite[:, None], table[:, 6:8], -1).astype(np.int32)
    return PersistenceBars(
        birth_prob=birth_prob,
        death_prob=death_prob,
        persistence=persistence,
        birth_rc=birth_rc,
        death_rc=death_rc,
        essential=~finite,
    )


def _compute_join_tree(prob: np.ndarray) -> PersistenceBars:
    """Reference 4-connected superlevel join tree.

    This is slower than cripser and is mainly used to verify the cripser bar
    multiset and to provide a dependency-free fallback for small tests.  It
    returns the same H0 ``PersistenceBars`` schema.
    """
    height, width = prob.shape
    flat = prob.ravel()
    order = np.argsort(-flat, kind='stable')
    parent = np.full(flat.size, -1, np.int64)
    birth = np.zeros(flat.size, np.float64)
    birth_idx = np.zeros(flat.size, np.int64)
    active = np.zeros(flat.size, bool)
    births, deaths, persistence, birth_rc, death_rc, essential = (
        [], [], [], [], [], [])

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != index:
            nxt = parent[index]
            parent[index] = root
            index = nxt
        return root

    def neighbours(index: int):
        row, col = divmod(index, width)
        if row > 0:
            yield index - width
        if row + 1 < height:
            yield index + width
        if col > 0:
            yield index - 1
        if col + 1 < width:
            yield index + 1

    for index in order:
        row, col = divmod(int(index), width)
        active[index] = True
        parent[index] = index
        birth[index] = flat[index]
        birth_idx[index] = index
        roots = {find(neighbour) for neighbour in neighbours(index)
                 if active[neighbour]}
        roots.discard(index)
        if not roots:
            continue
        survivor = max(roots, key=lambda root: (birth[root], -root))
        parent[index] = survivor
        for root in sorted(roots, key=lambda value: (birth[value], -value),
                           reverse=True):
            if root == survivor:
                continue
            parent[root] = survivor
            deaths.append(float(flat[index]))
            births.append(float(birth[root]))
            persistence.append(float(birth[root] - flat[index]))
            birth_rc.append(divmod(int(birth_idx[root]), width))
            death_rc.append((row, col))
            essential.append(False)

    roots = [index for index in range(flat.size)
             if active[index] and parent[index] == index]
    if roots:
        survivor = max(roots, key=lambda root: (birth[root], -root))
        births.append(float(birth[survivor]))
        deaths.append(0.0)
        persistence.append(float(birth[survivor]))
        birth_rc.append(divmod(int(birth_idx[survivor]), width))
        death_rc.append((-1, -1))
        essential.append(True)
    return PersistenceBars(
        birth_prob=np.asarray(births, np.float32),
        death_prob=np.asarray(deaths, np.float32),
        persistence=np.asarray(persistence, np.float32),
        birth_rc=np.asarray(birth_rc, np.int32).reshape(-1, 2),
        death_rc=np.asarray(death_rc, np.int32).reshape(-1, 2),
        essential=np.asarray(essential, bool),
    )


def _compute_gudhi(prob: np.ndarray) -> PersistenceBars:
    """MIT-licensed fallback; slower than cripser."""
    try:
        import gudhi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'A2SAM topology backend gudhi is not installed') from exc
    complex_ = gudhi.CubicalComplex(
        dimensions=list(prob.shape),
        top_dimensional_cells=(-prob).ravel(),
    )
    pairs = complex_.cofaces_of_persistence_pairs()
    finite_pairs = pairs[0][0] if pairs and pairs[0] else []
    essential_cells = pairs[1][0] if len(pairs) > 1 and pairs[1] else []
    width = prob.shape[1]
    births, deaths, persistence, birth_rc, death_rc, essential = (
        [], [], [], [], [], [])
    for birth_cell, death_cell in finite_pairs:
        br, bc = divmod(int(birth_cell), width)
        dr, dc = divmod(int(death_cell), width)
        bv = float(prob[br, bc])
        dv = float(prob[dr, dc])
        births.append(bv)
        deaths.append(dv)
        persistence.append(bv - dv)
        birth_rc.append((br, bc))
        death_rc.append((dr, dc))
        essential.append(False)
    for cell in essential_cells:
        br, bc = divmod(int(cell), width)
        bv = float(prob[br, bc])
        births.append(bv)
        deaths.append(0.0)
        persistence.append(bv)
        birth_rc.append((br, bc))
        death_rc.append((-1, -1))
        essential.append(True)
    return PersistenceBars(
        birth_prob=np.asarray(births, np.float32),
        death_prob=np.asarray(deaths, np.float32),
        persistence=np.asarray(persistence, np.float32),
        birth_rc=np.asarray(birth_rc, np.int32).reshape(-1, 2),
        death_rc=np.asarray(death_rc, np.int32).reshape(-1, 2),
        essential=np.asarray(essential, bool),
    )
