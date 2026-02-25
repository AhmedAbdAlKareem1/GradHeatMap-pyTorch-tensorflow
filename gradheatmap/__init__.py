from .core import HeatMap

try:
    from .coreTorch import HeatMapPyTorch
except Exception:
    HeatMapPyTorch = None

__all__ = ["HeatMap", "HeatMapPyTorch"]

