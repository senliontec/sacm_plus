"""Lazy adapter registry: model id -> self-contained adapter class."""

from __future__ import annotations

import importlib

_ADAPTERS = {
    'a2sam': ('baseline_zoo.adapters.a2sam', 'A2SAMAdapter'),
    'aco_moe_lora': ('baseline_zoo.adapters.aco_moe_lora',
                     'ACOMoeLoRAAdapter'),
    'bcunet': ('baseline_zoo.adapters.bcunet', 'BCUNetAdapter'),
    'blo_sam': ('baseline_zoo.adapters.blo_sam', 'BLOSAMAdapter'),
    'convlora': ('baseline_zoo.adapters.convlora', 'ConvLoRAAdapter'),
    'cs2net': ('baseline_zoo.adapters.cs2net', 'CS2NetAdapter'),
    'csunet': ('baseline_zoo.adapters.csunet', 'CSUNetAdapter'),
    'c2p': ('baseline_zoo.adapters.c2p', 'C2PAdapter'),
    'cwsam': ('baseline_zoo.adapters.cwsam', 'CWSAMAdapter'),
    'curvfield': ('baseline_zoo.adapters.curvfield', 'CurvFieldAdapter'),
    'dconnnet': ('baseline_zoo.adapters.dconnnet', 'DconnNetAdapter'),
    'dka': ('baseline_zoo.adapters.dka', 'DKAAdapter'),
    'dscnet': ('baseline_zoo.adapters.dscnet', 'DSCNetAdapter'),
    'flowsdf': ('baseline_zoo.adapters.flowsdf', 'FlowSDFAdapter'),
    'fssam': ('baseline_zoo.adapters.fssam', 'FSSAMAdapter'),
    'jtfn': ('baseline_zoo.adapters.jtfn', 'JTFNAdapter'),
    'knowsam': ('baseline_zoo.adapters.knowsam', 'KnowSAMAdapter'),
    'kprism': ('baseline_zoo.adapters.kprism', 'KPrismAdapter'),
    'liot': ('baseline_zoo.adapters.liot', 'LIOTAdapter'),
    'maskvsc': ('baseline_zoo.adapters.maskvsc', 'MaskVSCAdapter'),
    'matcher': ('baseline_zoo.adapters.matcher', 'MatcherAdapter'),
    'medcagd': ('baseline_zoo.adapters.medcagd', 'MedCAGDAdapter'),
    'nnunet2d': ('baseline_zoo.adapters.nnunet2d', 'NNUNet2DAdapter'),
    'nnwnet': ('baseline_zoo.adapters.nnwnet', 'NNWNetAdapter'),
    'persam': ('baseline_zoo.adapters.persam', 'PerSAMAdapter'),
    'rsrm': ('baseline_zoo.adapters.rsrm', 'RSRMAdapter'),
    'sam2_unet': ('baseline_zoo.adapters.sam2_unet', 'SAM2UNetAdapter'),
    'sam_adapter': ('baseline_zoo.adapters.sam_adapter', 'SAMAdapter'),
    'sam_med2d': ('baseline_zoo.adapters.sam_med2d', 'SamMed2DAdapter'),
    'sacm_official': ('baseline_zoo.adapters.sacm_official',
                      'SACMOfficialAdapter'),
    'sam_octa': ('baseline_zoo.adapters.sam_octa', 'SAMOCTAAdapter'),
    'sansa': ('baseline_zoo.adapters.sansa', 'SANSAAdapter'),
    's2mnet': ('baseline_zoo.adapters.s2mnet', 'S2MNetAdapter'),
    'sam_vit_l_unet': ('baseline_zoo.adapters.sam_vit_l_unet',
                       'SAMViTLUNetAdapter'),
    'scrwkv': ('baseline_zoo.adapters.scrwkv', 'SCRWKVAdapter'),
    'samed': ('baseline_zoo.adapters.samed', 'SAMedAdapter'),
    'segdino': ('baseline_zoo.adapters.segdino', 'SegDINOAdapter'),
    'segdino_v2': ('baseline_zoo.adapters.segdino_v2', 'SegDINOV2Adapter'),
    'topolora': ('baseline_zoo.adapters.topolora', 'TopoLoRAAdapter'),
    'tpsegformer': ('baseline_zoo.adapters.tpsegformer',
                    'TPSegformerAdapter'),
    'ucs': ('baseline_zoo.adapters.ucs', 'UCSAdapter'),
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
