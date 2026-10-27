"""Shared 3-shot support-set selection for few-shot/training-free baselines.

The project trains on 18 images (3 per seen domain) and evaluates on the
20-dataset protocol.  A support-based method needs labeled support images for
each test dataset.  Seen datasets use their own three training shots; unseen
and novel datasets use a structurally matched donor domain.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from baseline_zoo.data import Sample, load_mask, load_split_samples

CRACK_LIKE = {
    'DeepCrack', 'CHCrack5k', 'TopoMortar', 'TYRE', 'WIRE'}
VESSEL_LIKE = {
    'FIVES', 'STARE', 'XCAD', 'HRF', 'LES-AV', 'OCTA-500', 'ROAD',
    'LEAF', 'DeepGlobe'}
DONOR_PREFERENCE = (
    'DRIVE', 'CHASEDB1', 'DCA1', 'CREMI', 'CrackTree200', 'CORN')


def load_support_groups(data_root: str | Path) -> dict[str, list[Sample]]:
    """Load the 18-image train split and group it by dataset."""
    samples = load_split_samples(Path(data_root), 'train')
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.dataset].append(sample)
    for values in groups.values():
        values.sort(key=lambda item: item.stem)
    return dict(groups)


def choose_support_dataset(name: str, groups: dict[str, list[Sample]],
                           override: str = 'auto') -> str:
    """Pick the support domain for ``name`` (same domain or a donor)."""
    if override and override != 'auto':
        if override not in groups:
            raise SystemExit(
                f'support dataset {override!r} is not in the training split; '
                f'available: {sorted(groups)}')
        return override
    if name in groups:
        return name
    preferred: list[str] = []
    if name in CRACK_LIKE:
        preferred.append('CrackTree200')
    elif name in VESSEL_LIKE:
        preferred.append('DRIVE')
    preferred.extend(DONOR_PREFERENCE)
    for candidate in preferred:
        if candidate in groups:
            return candidate
    if not groups:
        raise SystemExit('the training split has no support samples')
    return sorted(groups)[0]


def select_supports(
    name: str,
    groups: dict[str, list[Sample]],
    shots: int = 3,
    override: str = 'auto',
    require_nonempty: bool = True,
) -> tuple[list[Sample], str]:
    """Return up to ``shots`` support samples and the donor domain name."""
    donor = choose_support_dataset(name, groups, override=override)
    pairs: Iterable[Sample] = groups[donor]
    if require_nonempty:
        pairs = [sample for sample in pairs
                 if load_mask(sample.mask_path).any()]
    pairs = list(pairs)[:max(int(shots), 1)]
    if not pairs:
        raise RuntimeError(f'no non-empty support mask in {donor}')
    return pairs, donor


def mask_is_nonempty(sample: Sample) -> bool:
    return bool(load_mask(sample.mask_path).any())


def dataset_names(groups: dict[str, list[Sample]]) -> Optional[list[str]]:
    return sorted(groups) if groups else None
