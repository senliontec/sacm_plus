"""CoSNA baseline adapter.

The implementation lives in the self-contained
``baseline_zoo.models.cosna`` package.  This module only exposes the adapter
class to the lazy registry so that ``baseline_zoo.run_one --model cosna``
works without importing any other model package.
"""

from baseline_zoo.models.cosna.adapter import CoSNAAdapter

__all__ = ['CoSNAAdapter']
