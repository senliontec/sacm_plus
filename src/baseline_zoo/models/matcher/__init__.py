"""Self-contained Matcher (DINOv2 + SAM) model package."""

from .model import MatcherModel, build_matcher

__all__ = ['MatcherModel', 'build_matcher']
