"""Lazy adapter registry: model id -> self-contained adapter class."""

from __future__ import annotations

import importlib

_ADAPTERS = {
    'bcunet': ('baseline_zoo.adapters.bcunet', 'BCUNetAdapter'),
    'cs2net': ('baseline_zoo.adapters.cs2net', 'CS2NetAdapter'),
    'cwsam': ('baseline_zoo.adapters.cwsam', 'CWSAMAdapter'),
    'fob_sam': ('baseline_zoo.adapters.fob_sam', 'FoBSAMAdapter'),
    'knowsam': ('baseline_zoo.adapters.knowsam', 'KnowSAMAdapter'),
    'maskvsc': ('baseline_zoo.adapters.maskvsc', 'MaskVSCAdapter'),
    'nnwnet': ('baseline_zoo.adapters.nnwnet', 'NNWNetAdapter'),
    'sam_med2d': ('baseline_zoo.adapters.sam_med2d', 'SamMed2DAdapter'),
    'sacm_official': ('baseline_zoo.adapters.sacm_official',
                      'SACMOfficialAdapter'),
    'sam_octa': ('baseline_zoo.adapters.sam_octa', 'SAMOCTAAdapter'),
    'sam_vit_l_unet': ('baseline_zoo.adapters.sam_vit_l_unet',
                       'SAMViTLUNetAdapter'),
    'segdino': ('baseline_zoo.adapters.segdino', 'SegDINOAdapter'),
    'unet': ('baseline_zoo.adapters.unet', 'UNetAdapter'),
}


def available_adapters() -> list[str]:
    return sorted(_ADAPTERS)


def load_adapter(model_id: str):
    try:
        module_name, class_name = _ADAPTERS[model_id]
    except KeyError:
        raise KeyError(
            f'no unified adapter for {model_id!r}; '
            f'ported adapters: {", ".join(available_adapters())}') from None
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()
