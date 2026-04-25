"""Backward-compatible re-export of the TensorFlow HeatMap class.

Import from the package root instead:
    from gradheatmap import HeatMap
"""
from __future__ import annotations

from .tf_heatmap import HeatMap

__all__ = ["HeatMap"]
