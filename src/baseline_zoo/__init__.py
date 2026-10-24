"""Unified model registry and experiment planning for the paper pipeline."""

from .protocols import ProtocolConfig, SeedSplit, load_protocol, load_protocols
from .registry import ModelConfig, load_models

__all__ = [
    'ModelConfig',
    'ProtocolConfig',
    'SeedSplit',
    'load_models',
    'load_protocol',
    'load_protocols',
]
__version__ = '0.1.0'
