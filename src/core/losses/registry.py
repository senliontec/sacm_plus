"""Topology-loss registry: name -> loss class, with graceful registration.

Importing any loss module registers its CLI-capable entries here. Losses
whose hard dependencies are missing simply skip registration, so the
package always imports cleanly.
"""

TOPOLOGY_LOSS_REGISTRY = {}


def register(name):
    def _wrap(cls):
        TOPOLOGY_LOSS_REGISTRY[name] = cls
        return cls
    return _wrap


def build_topology_loss(name, **kwargs):
    """Instantiate a registered topology loss by name; None when 'none'."""
    if name is None or name == 'none':
        return None
    if name not in TOPOLOGY_LOSS_REGISTRY:
        raise ValueError(
            f"Unknown topology loss {name!r}; available: "
            f"{sorted(TOPOLOGY_LOSS_REGISTRY)}"
        )
    return TOPOLOGY_LOSS_REGISTRY[name](**kwargs)
