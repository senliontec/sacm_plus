"""Topology-loss monitor: evaluate EVERY registered topology loss on the
current predictions each validation epoch.

Monitoring only — these values never enter the training objective; they
show which topology dimension (endpoint connectivity, persistence
matching, component-graph structure, ...) is hurting, so the model
improvement direction can be read off the log.

Engine-based losses (betti family / satloss / topograph family) run at
`resolution` (128 default — cheap enough for per-epoch monitoring on the
tiny few-shot val set); pure-torch losses run at the input resolution.
"""

import logging

import torch

from .registry import TOPOLOGY_LOSS_REGISTRY

_monitor_cache = {}


def topology_loss_monitor(logits, target, resolution=128):
    """Return {name: value} for every registered topology loss.

    Values are floats; a loss that fails to run yields NaN (displayed as
    '—') without breaking the monitoring loop.
    """
    # 适配器契约是 [B,1,H,W];验证循环常传 2D,这里统一归一化
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    out = {}
    for name in sorted(TOPOLOGY_LOSS_REGISTRY):
        try:
            if name not in _monitor_cache:
                try:
                    loss = TOPOLOGY_LOSS_REGISTRY[name](resolution=resolution)
                except TypeError:
                    loss = TOPOLOGY_LOSS_REGISTRY[name]()
                _monitor_cache[name] = loss
            with torch.no_grad():
                v = _monitor_cache[name](logits, target)
            out[name] = float(v.item() if torch.is_tensor(v) else v)
        except Exception as e:
            out[name] = float('nan')
            logging.debug(f"topology monitor {name} failed: {e}")
    return out
