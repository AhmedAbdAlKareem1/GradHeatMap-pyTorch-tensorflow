"""TensorFlow Grad-CAM tests using small in-memory dummy models.

No pretrained weights are downloaded — all tests run offline.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — tiny models built entirely from scratch
# ---------------------------------------------------------------------------

def _binary_model():
    tf = pytest.importorskip("tensorflow")
    inp = tf.keras.Input((32, 32, 3))
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)


def _binary_softmax_model():
    tf = pytest.importorskip("tensorflow")
    inp = tf.keras.Input((32, 32, 3))
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def _multiclass_model(num_classes: int = 4):
    tf = pytest.importorskip("tensorflow")
    inp = tf.keras.Input((32, 32, 3))
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def _model_with_rescaling():
    """Model with internal Rescaling — raw uint8 pixels should be passed."""
    tf = pytest.importorskip("tensorflow")
    inp = tf.keras.Input((32, 32, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inp)
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inp, out)


# ---------------------------------------------------------------------------
# Tests — binary sigmoid
# ---------------------------------------------------------------------------

class TestTFBinarySigmoid:
    def test_overlay_returns_correct_shape(self, dummy_image_path):
        pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap

        hm = HeatMap(model=_binary_model(), img_path=dummy_image_path, class_names=["neg", "pos"])
        overlay = hm.overlay_heatmap(alpha=0.4)

        assert overlay is not None
        assert overlay.ndim == 3
        assert overlay.shape[2] == 3
        assert overlay.shape[:2] == (64, 64)

    def test_compute_gradcam_is_2d_normalized(self, dummy_image_path):
        pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap

        hm = HeatMap(model=_binary_model(), img_path=dummy_image_path, class_names=["neg", "pos"])
        cam = hm.compute_gradcam()

        assert isinstance(cam, np.ndarray)
        assert cam.ndim == 2
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0 + 1e-5

    def test_explicit_target_class_0(self, dummy_image_path):
        pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap

        hm = HeatMap(model=_binary_model(), img_path=dummy_image_path,
                     class_names=["neg", "pos"], target_class=0)
        hm.overlay_heatmap()   # must not raise

    def test_explicit_target_class_1(self, dummy_image_path):
        pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap

        hm = HeatMap(model=_binary_model(), img_path=dummy_image_path,
                     class_names=["neg", "pos"], target_class=1)
        hm.overlay_heatmap()

    def test_save_heat_img_creates_file(self, dummy_image_path, tmp_path, monkeypatch):
        pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap

        monkeypatch.chdir(tmp_path)
        hm = HeatMap(model=_binary_model(), img_path=dummy_image_path, class_names=["neg", "pos"])
        overlay = hm.overlay_heatmap()
        hm.save_heat_img("result", overlay)   # no extension → .jpg appended

        assert os.path.isfile(tmp_path / "heatmap" / "result.jpg")

    def test_save_heat_img_with_extension(self, dummy_image_path, tmp_path, monkeypatch):
        pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap

        monkeypatch.chdir(tmp_path)
        hm = HeatMap(model=_binary_model(), img_path=dummy_image_path, class_names=["neg", "pos"])
        hm.save_heat_img("out.jpg", hm.overlay_heatmap())
        assert os.path.isfile(tmp_path / "heatmap" / "out.jpg")


# ---------------------------------------------------------------------------
# Tests — binary softmax (2-class)
# ---------------------------------------------------------------------------

class TestTFBinarySoftmax:
    def test_overlay_shape(self, dummy_image_path):
        pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap

        hm = HeatMap(model=_binary_softmax_model(), img_path=dummy_image_path,
                     class_names=["no", "yes"])
        overlay = hm.overlay_heatmap()
        assert overlay.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Tests — multiclass
# ---------------------------------------------------------------------------

class TestTFMulticlass:
    def test_overlay_shape(self, dummy_image_path):
        pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap

        hm = HeatMap(model=_multiclass_model(4), img_path=dummy_image_path,
                     class_names=["a", "b", "c", "d"])
        overlay = hm.overlay_heatmap()
        assert overlay.shape == (64, 64, 3)

    def test_target_class_explicit(self, dummy_image_path):
        pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap

        hm = HeatMap(model=_multiclass_model(4), img_path=dummy_image_path,
                     class_names=["a", "b", "c", "d"], target_class=2)
        hm.overlay_heatmap()


# ---------------------------------------------------------------------------
# Tests — internal Rescaling layer
# ---------------------------------------------------------------------------

class TestTFRescalingModel:
    def test_rescaling_model(self, dummy_image_path):
        pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap

        hm = HeatMap(model=_model_with_rescaling(), img_path=dummy_image_path,
                     class_names=["x", "y"])
        hm.overlay_heatmap()  # must not raise


# ---------------------------------------------------------------------------
# Tests — error handling
# ---------------------------------------------------------------------------

class TestTFErrors:
    def test_bad_image_path_raises(self, tmp_path):
        pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap

        hm = HeatMap(model=_binary_model(), img_path="/no/such/file.jpg",
                     class_names=["neg", "pos"])
        with pytest.raises(FileNotFoundError):
            hm.overlay_heatmap()

    def test_model_loaded_from_path(self, tmp_path, dummy_image_path):
        tf = pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap

        model_path = str(tmp_path / "m.keras")
        _binary_model().save(model_path)

        hm = HeatMap(model=model_path, img_path=dummy_image_path, class_names=["a", "b"])
        hm.overlay_heatmap()

    def test_version_exported(self):
        import gradheatmap
        assert hasattr(gradheatmap, "__version__")
        assert gradheatmap.__version__

    def test_user_callable_preprocess(self, dummy_image_path):
        pytest.importorskip("tensorflow")
        from gradheatmap import HeatMap
        import numpy as np

        def my_prep(img):
            return img.astype(np.float32) / 127.5 - 1.0

        hm = HeatMap(model=_binary_model(), img_path=dummy_image_path,
                     class_names=["neg", "pos"], preprocess=my_prep)
        hm.overlay_heatmap()
