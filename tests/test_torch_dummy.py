"""PyTorch Grad-CAM tests using small in-memory dummy models.

No pretrained weights are downloaded — all tests run offline.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — tiny models
# ---------------------------------------------------------------------------

def _binary_cnn():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


def _binary_softmax_cnn():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 2),
        nn.Softmax(dim=1),
    )


def _multiclass_cnn(num_classes: int = 3):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, num_classes),
        nn.Softmax(dim=1),
    )


def _grayscale_cnn():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 2),
        nn.Softmax(dim=1),
    )


# ---------------------------------------------------------------------------
# Tests — binary sigmoid
# ---------------------------------------------------------------------------

class TestTorchBinarySigmoid:
    def test_overlay_returns_correct_shape(self, dummy_image_path):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        hm = HeatMapPyTorch(model=_binary_cnn(), img_path=dummy_image_path,
                             class_names=["neg", "pos"], image_size=(32, 32))
        overlay = hm.overlay_heatmap(alpha=0.4)

        assert overlay is not None
        assert overlay.shape == (64, 64, 3)

    def test_compute_gradcam_is_2d(self, dummy_image_path):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        hm = HeatMapPyTorch(model=_binary_cnn(), img_path=dummy_image_path,
                             class_names=["neg", "pos"], image_size=(32, 32))
        cam = hm.compute_gradcam()
        assert isinstance(cam, np.ndarray)
        assert cam.ndim == 2
        assert cam.min() >= 0.0

    def test_explicit_target_class(self, dummy_image_path):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        for tc in (0, 1):
            hm = HeatMapPyTorch(model=_binary_cnn(), img_path=dummy_image_path,
                                 class_names=["neg", "pos"], target_class=tc,
                                 image_size=(32, 32))
            hm.overlay_heatmap()  # must not raise

    def test_save_heat_img_creates_file(self, dummy_image_path, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        monkeypatch.chdir(tmp_path)
        hm = HeatMapPyTorch(model=_binary_cnn(), img_path=dummy_image_path,
                             class_names=["neg", "pos"], image_size=(32, 32))
        hm.save_heat_img("result", hm.overlay_heatmap())
        assert os.path.isfile(tmp_path / "heatmap" / "result.jpg")

    def test_save_heat_img_custom_folder(self, dummy_image_path, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        monkeypatch.chdir(tmp_path)
        hm = HeatMapPyTorch(model=_binary_cnn(), img_path=dummy_image_path,
                             class_names=["neg", "pos"], image_size=(32, 32))
        hm.save_heat_img("out.jpg", hm.overlay_heatmap(), folder="my_output")
        assert os.path.isfile(tmp_path / "my_output" / "out.jpg")


# ---------------------------------------------------------------------------
# Tests — binary softmax (2-class)
# ---------------------------------------------------------------------------

class TestTorchBinarySoftmax:
    def test_overlay_shape(self, dummy_image_path):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        hm = HeatMapPyTorch(model=_binary_softmax_cnn(), img_path=dummy_image_path,
                             class_names=["no", "yes"], image_size=(32, 32))
        overlay = hm.overlay_heatmap()
        assert overlay.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Tests — multiclass
# ---------------------------------------------------------------------------

class TestTorchMulticlass:
    def test_overlay_shape(self, dummy_image_path):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        hm = HeatMapPyTorch(model=_multiclass_cnn(3), img_path=dummy_image_path,
                             class_names=["a", "b", "c"], image_size=(32, 32))
        overlay = hm.overlay_heatmap()
        assert overlay.shape == (64, 64, 3)

    def test_target_class_explicit(self, dummy_image_path):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        hm = HeatMapPyTorch(model=_multiclass_cnn(3), img_path=dummy_image_path,
                             class_names=["a", "b", "c"], target_class=1,
                             image_size=(32, 32))
        hm.overlay_heatmap()


# ---------------------------------------------------------------------------
# Tests — grayscale input
# ---------------------------------------------------------------------------

class TestTorchGrayscale:
    def test_grayscale_model(self, dummy_image_path):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        hm = HeatMapPyTorch(model=_grayscale_cnn(), img_path=dummy_image_path,
                             class_names=["x", "y"], image_size=(32, 32))
        overlay = hm.overlay_heatmap()
        assert overlay.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Tests — custom preprocessing
# ---------------------------------------------------------------------------

class TestTorchPreprocess:
    def test_imagenet_preprocess_string(self, dummy_image_path):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        hm = HeatMapPyTorch(model=_binary_cnn(), img_path=dummy_image_path,
                             class_names=["neg", "pos"], preprocess="resnet50",
                             image_size=(32, 32))
        hm.overlay_heatmap()

    def test_user_callable_preprocess(self, dummy_image_path):
        pytest.importorskip("torch")
        import torch
        from gradheatmap import HeatMapPyTorch

        def my_prep(img_rgb: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        hm = HeatMapPyTorch(model=_binary_cnn(), img_path=dummy_image_path,
                             class_names=["neg", "pos"], preprocess=my_prep,
                             image_size=(32, 32))
        hm.overlay_heatmap()

    def test_invalid_preprocess_string_raises(self, dummy_image_path):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        hm = HeatMapPyTorch(model=_binary_cnn(), img_path=dummy_image_path,
                             class_names=["neg", "pos"], preprocess="unknownnet",
                             image_size=(32, 32))
        with pytest.raises(ValueError, match="Unknown preprocess"):
            hm.overlay_heatmap()


# ---------------------------------------------------------------------------
# Tests — error handling
# ---------------------------------------------------------------------------

class TestTorchErrors:
    def test_string_model_raises(self):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        with pytest.raises(ValueError, match="torch.nn.Module"):
            HeatMapPyTorch(model="path/to/model.pth", img_path="x.jpg", class_names=[])

    def test_missing_image_raises(self):
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        hm = HeatMapPyTorch(model=_binary_cnn(), img_path="/nonexistent/img.jpg",
                             class_names=["a", "b"])
        with pytest.raises(FileNotFoundError):
            hm.overlay_heatmap()

    def test_object_reuse(self, dummy_image_path):
        """Calling overlay_heatmap twice on the same instance must not crash."""
        pytest.importorskip("torch")
        from gradheatmap import HeatMapPyTorch

        hm = HeatMapPyTorch(model=_binary_cnn(), img_path=dummy_image_path,
                             class_names=["neg", "pos"], image_size=(32, 32))
        hm.overlay_heatmap()
        hm.overlay_heatmap()   # second call should also succeed
