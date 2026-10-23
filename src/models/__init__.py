"""Model implementations: one module per segmentation model/family.

Importing a model module registers its ModelSpecs into the framework's
MODEL_REGISTRY (core.registry) and exposes the model-specific
configuration (presets, diagnostics).

Current models:
  - sam_network: the SAM network implementation (Meta code vendored,
    with the SACM geometric adapters / closed-loop decoder switches).
  - sacm_x: the SACM-X algorithm layer (prompt-free curvilinear
    segmentation: dual-level geometric adapters + closed-loop
    coarse-to-fine decoder + topology-aligned training); registers
    sam_l / sam_b / sam_h.

Adding a new model: create models/<name>/ with the architecture code, a
models.py registering its ModelSpecs, and its own configs; then add
`from . import <name>` below. The framework scripts pick it up
automatically via the shared MODEL_REGISTRY.
"""

from . import sacm_x  # noqa: F401  (registers sam_l/sam_b/sam_h)
