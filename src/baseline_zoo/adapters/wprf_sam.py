"""WPRF-SAM baseline adapter.

The implementation lives in the self-contained
``baseline_zoo.models.wprf_sam`` package.  This module only exposes the
adapter class to the lazy registry so that
``baseline_zoo.run_one --model wprf_sam`` works without importing any other
model package.
"""

from baseline_zoo.models.wprf_sam.wprf_adapter import WPRFAdapter

__all__ = ['WPRFAdapter']
