"""
gradheatmap
===========
Automatic Grad-CAM heatmap generation for TensorFlow/Keras and PyTorch models.

Quick start
-----------
TensorFlow / Keras::

    from gradheatmap import HeatMap

    hm = HeatMap(model="my_model.keras", img_path="image.jpg", class_names=["cat", "dog"])
    hm.save_heat_img("result.jpg", hm.overlay_heatmap())

PyTorch::

    from gradheatmap import HeatMapPyTorch

    hm = HeatMapPyTorch(model=my_model, img_path="image.jpg",
                         class_names=["cat", "dog"], preprocess="resnet50")
    hm.save_heat_img("result.jpg", hm.overlay_heatmap())
"""
from __future__ import annotations

from typing import Any

__version__ = "0.2.1"


class HeatMap:  # pragma: no cover
    """
    Lazy proxy for the TensorFlow/Keras Grad-CAM backend.

    Importing ``gradheatmap`` does **not** require TensorFlow to be installed.
    TensorFlow is only imported when you instantiate this class.

    Install TensorFlow support with::

        pip install "gradheatmap[tf]"

    See :class:`gradheatmap.tf_heatmap.HeatMap` for the full API.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "HeatMap":
        try:
            from .tf_heatmap import HeatMap as _HeatMap
        except ImportError as e:
            raise ImportError(
                "TensorFlow backend is not available. "
                'Install it with:  pip install "gradheatmap[tf]"'
            ) from e
        return _HeatMap(*args, **kwargs)


class HeatMapPyTorch:  # pragma: no cover
    """
    Lazy proxy for the PyTorch Grad-CAM backend.

    Importing ``gradheatmap`` does **not** require PyTorch to be installed.
    PyTorch is only imported when you instantiate this class.

    Install PyTorch support with::

        pip install "gradheatmap[torch]"

    See :class:`gradheatmap.coreTorch.HeatMapPyTorch` for the full API.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "HeatMapPyTorch":
        try:
            from .coreTorch import HeatMapPyTorch as _HeatMapPyTorch
        except ImportError as e:
            raise ImportError(
                "PyTorch backend is not available. "
                'Install it with:  pip install "gradheatmap[torch]"'
            ) from e
        return _HeatMapPyTorch(*args, **kwargs)


__all__ = ["HeatMap", "HeatMapPyTorch", "__version__"]
