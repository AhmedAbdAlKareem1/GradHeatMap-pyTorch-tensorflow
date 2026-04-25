"""TensorFlow/Keras Grad-CAM implementation."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from ._utils import build_overlay, load_image_bgr, save_heatmap_image


# ---------------------------------------------------------------------------
# Helper layer — strips the final activation so Grad-CAM uses raw logits
# ---------------------------------------------------------------------------

class _LinearDense(tf.keras.layers.Layer):
    """Dense layer that shares weights with *source_layer* but applies no activation."""

    def __init__(self, source_layer: tf.keras.layers.Dense, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.w = source_layer.kernel
        self.b = source_layer.bias

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.matmul(inputs, self.w) + self.b


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class HeatMap:
    """
    Grad-CAM heatmap generator for TensorFlow/Keras classification models.

    Supports:
    - Binary classification (sigmoid, single output)
    - Binary classification (softmax, two-class output)
    - Multiclass classification (softmax)
    - Custom and transfer-learning models (nested backbone detection)
    - Smart preprocessing: auto-detects Rescaling layers, backbone normalization,
      or falls back to [0, 1] scaling

    Parameters
    ----------
    model:
        A ``tf.keras.Model`` instance **or** a path to a ``.keras`` / ``.h5`` file.
    img_path:
        Path to the input image.
    class_names:
        List of human-readable class labels.
    target_class:
        Class index to explain.  ``None`` (default) uses the predicted class.
    preprocess:
        Preprocessing override.  Options:

        - ``None`` (default) — auto-detect from backbone or Rescaling layer
        - A callable ``fn(img_uint8_rgb) -> np.ndarray`` — your own normalizer
        - A string name such as ``"resnet50"``, ``"vgg16"``, ``"efficientnet"``
    """

    _PREPROCESS_MAP: Dict[str, Any] = {}  # populated lazily to avoid top-level TF imports

    def __init__(
        self,
        model: Union[tf.keras.Model, str],
        img_path: str,
        class_names: List[str],
        target_class: Optional[int] = None,
        preprocess: Optional[Union[str, Any]] = None,
    ) -> None:
        self.model = (
            model
            if isinstance(model, tf.keras.Model)
            else load_model(model, compile=False)
        )
        self.user_preprocess = preprocess
        self.img_path = img_path
        self.class_names = class_names
        self.target_class = target_class

        # Backbone detection
        self.pre_trained_model = self._detect_backbone()
        if self.pre_trained_model is None:
            self.pre_trained_model = self.model
            self.backbone_name: Optional[str] = None
        else:
            self.backbone_name = self.pre_trained_model.name.lower()

        # Image size
        self.image_size: Tuple[int, int] = self._extract_image_size()

        # Monitor mode tracking (used by experiment_saver integration, no-op here)
        print(f"Detected image size   : {self.image_size}")
        print(f"Detected backbone     : {self.backbone_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_gradcam(self) -> np.ndarray:
        """
        Run Grad-CAM and return the raw 2-D heatmap.

        Returns
        -------
        np.ndarray
            2-D float array with values in [0, 1], shape (H_cam, W_cam).
        """
        x = self.preprocess_image()
        idx, _name, _conf, _probs = self.predict(x)
        grad_model = self.build_grad_model()

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)

            if predictions.shape[-1] == 1:
                explain_class = int(self.target_class) if self.target_class is not None else int(idx)
                raw = predictions[:, 0]
                # Use log to get stable gradients; treat as binary sigmoid logit
                loss = tf.math.log(raw + 1e-10) if explain_class == 1 else tf.math.log(1.0 - raw + 1e-10)
            else:
                class_index = (
                    int(self.target_class)
                    if self.target_class is not None
                    else int(tf.argmax(predictions[0]))
                )
                loss = tf.math.log(predictions[:, class_index] + 1e-10)

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            raise ValueError(
                "Gradients are None — the targeted conv layer may be disconnected "
                "from the loss in the grad model. Check build_grad_model() connectivity."
            )

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        acts = conv_outputs.numpy()[0]        # (H, W, C)
        pg = pooled_grads.numpy()             # (C,)

        acts *= pg[np.newaxis, np.newaxis, :]  # broadcast weight
        heatmap = np.mean(acts, axis=-1)       # (H, W)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-10)
        return heatmap

    def overlay_heatmap(self, alpha: float = 0.4) -> np.ndarray:
        """
        Compute Grad-CAM and return a BGR overlay image.

        Parameters
        ----------
        alpha:
            Weight of the original image (0 = heatmap only, 1 = original only).

        Returns
        -------
        np.ndarray
            BGR uint8 image with the heatmap blended over the original.
        """
        heatmap = self.compute_gradcam()
        img_bgr = load_image_bgr(self.img_path)
        return build_overlay(heatmap, img_bgr, alpha=alpha)

    def save_heat_img(self, name: str, output_img: np.ndarray, folder: str = "heatmap") -> str:
        """
        Save *output_img* to ``{folder}/{name}``.

        Parameters
        ----------
        name:
            Filename.  ``.jpg`` is appended if no extension is given.
        output_img:
            BGR image returned by :meth:`overlay_heatmap`.
        folder:
            Output directory (default ``"heatmap"``).

        Returns
        -------
        str
            Full path written.
        """
        return save_heatmap_image(output_img, name, folder=folder)

    # ------------------------------------------------------------------
    # Backbone & layer detection
    # ------------------------------------------------------------------

    def _detect_backbone(self) -> Optional[tf.keras.Model]:
        """Return the largest nested sub-model, or None for flat models."""
        candidates = [l for l in self.model.layers if isinstance(l, tf.keras.Model)]
        if not candidates:
            print(f"No nested backbone found. Using top model: {self.model.name}")
            return None
        backbone = max(candidates, key=lambda m: m.count_params())
        print(f"Detected backbone: {backbone.name}")
        return backbone

    def _extract_image_size(self) -> Tuple[int, int]:
        """Extract (height, width) from the model's input shape."""
        input_shape = (
            self.model.input_shape[0]
            if isinstance(self.model.input_shape, list)
            else self.model.input_shape
        )
        h, w = input_shape[1], input_shape[2]
        if h is None or w is None:
            raise ValueError(
                "Model has a dynamic input size (None in height or width). "
                "This is not yet supported — build the model with a fixed input shape."
            )
        return int(h), int(w)

    def last_conv_layer_get(self, model: tf.keras.Model) -> str:
        """Return the name of the last Conv2D / DepthwiseConv2D in *model*."""
        def _find(m: tf.keras.Model) -> Optional[str]:
            for lyr in reversed(m.layers):
                if isinstance(lyr, tf.keras.Model):
                    found = _find(lyr)
                    if found is not None:
                        return found
                if isinstance(lyr, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    return lyr.name
            return None

        name = _find(model)
        if name is None:
            raise ValueError(
                f"No Conv2D or DepthwiseConv2D layer found in '{model.name}'. "
                "Grad-CAM requires at least one convolutional layer."
            )
        print(f"Last conv layer: {name}")
        return name

    # ------------------------------------------------------------------
    # Grad model construction
    # ------------------------------------------------------------------

    def build_grad_model(self) -> tf.keras.Model:
        """Build a Keras model that outputs (conv_activations, logits)."""
        conv_layer_name = self.last_conv_layer_get(self.pre_trained_model)
        last_layer = self.model.layers[-1]
        has_activation = (
            hasattr(last_layer, "activation")
            and last_layer.activation != tf.keras.activations.linear
        )

        # ── CASE 1: Nested backbone (VGG16, ResNet, etc.) ───────────────
        if self.pre_trained_model is not self.model:
            backbone_grad_model = tf.keras.models.Model(
                inputs=self.pre_trained_model.inputs,
                outputs=[
                    self.pre_trained_model.get_layer(conv_layer_name).output,
                    self.pre_trained_model.output,
                ],
            )

            outer_input = self.model.inputs
            conv_out, backbone_out = backbone_grad_model(outer_input)

            x = backbone_out
            head_started = False
            for layer in self.model.layers:
                if layer is self.pre_trained_model:
                    head_started = True
                    continue
                if head_started:
                    if layer is last_layer and has_activation:
                        x = _LinearDense(last_layer)(x)
                    else:
                        x = layer(x)

            return tf.keras.models.Model(inputs=outer_input, outputs=[conv_out, x])

        # ── CASE 2: Flat model (custom CNN, no nested backbone) ──────────
        if has_activation:
            # Use the input tensor of the last layer — works for any architecture
            logit_output = _LinearDense(last_layer)(last_layer.input)
        else:
            logit_output = self.model.output

        return tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(conv_layer_name).output, logit_output],
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def get_preprocess_function(self) -> Optional[Any]:
        """
        Resolve the preprocessing function.

        Priority:
        1. User-supplied callable
        2. User-supplied string name
        3. Auto-detected from backbone name
        4. Internal ``Rescaling`` layer detected → return None (raw pixels)
        5. Default: None (will scale to [0, 1] manually)
        """
        mapping = {
            "vgg16":           tf.keras.applications.vgg16.preprocess_input,
            "vgg19":           tf.keras.applications.vgg19.preprocess_input,
            "resnet":          tf.keras.applications.resnet.preprocess_input,
            "resnet50":        tf.keras.applications.resnet.preprocess_input,
            "resnetv2":        tf.keras.applications.resnet_v2.preprocess_input,
            "mobilenet":       tf.keras.applications.mobilenet.preprocess_input,
            "mobilenetv2":     tf.keras.applications.mobilenet_v2.preprocess_input,
            "mobilenetv3":     tf.keras.applications.mobilenet_v3.preprocess_input,
            "efficientnet":    tf.keras.applications.efficientnet.preprocess_input,
            "efficientnetv2":  tf.keras.applications.efficientnet_v2.preprocess_input,
            "densenet":        tf.keras.applications.densenet.preprocess_input,
            "inception":       tf.keras.applications.inception_v3.preprocess_input,
            "inceptionresnet": tf.keras.applications.inception_resnet_v2.preprocess_input,
            "xception":        tf.keras.applications.xception.preprocess_input,
            "nasnet":          tf.keras.applications.nasnet.preprocess_input,
            "convnext":        tf.keras.applications.convnext.preprocess_input,
            "regnet":          tf.keras.applications.regnet.preprocess_input,
        }

        p = self.user_preprocess

        if callable(p):
            print("Using user-supplied preprocessing callable.")
            return p

        if isinstance(p, str):
            key = p.lower().strip()
            fn = mapping.get(key)
            if fn is None:
                raise ValueError(
                    f"Unknown preprocess='{p}'. "
                    f"Pass a callable or one of: {sorted(mapping.keys())}"
                )
            print(f"Using user-specified preprocessing: '{p}'.")
            return fn

        if self.backbone_name:
            key = self.backbone_name.lower().strip()
            fn = mapping.get(key)
            if fn is not None:
                print(f"Auto-detected backbone preprocessing: '{self.backbone_name}'.")
                return fn
            for k, v in mapping.items():
                if k in key:
                    print(f"Auto-matched backbone preprocessing '{k}' from '{self.backbone_name}'.")
                    return v

        return None

    def preprocess_image(self) -> np.ndarray:
        """Load, resize, and normalize the image for model inference."""
        img = load_image_bgr(self.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.resize dsize = (width, height); self.image_size = (H, W)
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        img = np.expand_dims(img, axis=0)  # (1, H, W, 3)

        fn = self.get_preprocess_function()
        if fn is not None:
            return fn(img)

        # Check for an internal Rescaling layer (e.g. added during augmentation setup)
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Rescaling):
                print(
                    f"Internal Rescaling layer detected (scale={layer.scale}, "
                    f"offset={layer.offset}). Passing raw uint8 pixels."
                )
                return img  # model handles normalization internally

        print("No backbone or Rescaling layer found — scaling pixels to [0, 1].")
        return img.astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, img: np.ndarray
    ) -> Tuple[int, str, float, Dict[str, float]]:
        """
        Run a forward pass and return (class_index, class_name, confidence, prob_dict).

        Handles:
        - Sigmoid single output  (shape ``(1, 1)``)
        - Softmax two-class      (shape ``(1, 2)``)
        - Softmax multiclass     (shape ``(1, C)``)
        """
        preds = self.model(img, training=False).numpy()[0]

        if preds.shape[-1] == 1:
            score = float(preds[0])
            idx = 1 if score >= 0.5 else 0
            name = self.class_names[idx] if idx < len(self.class_names) else str(idx)
            conf = score if idx == 1 else 1.0 - score
            probs: Dict[str, float] = {
                (self.class_names[0] if len(self.class_names) > 0 else "0"): round(1.0 - score, 4),
                (self.class_names[1] if len(self.class_names) > 1 else "1"): round(score, 4),
            }
        else:
            idx = int(np.argmax(preds))
            conf = float(preds[idx])
            probs = {
                (self.class_names[i] if i < len(self.class_names) else str(i)): round(float(p), 4)
                for i, p in enumerate(preds)
            }
            name = self.class_names[idx] if idx < len(self.class_names) else str(idx)

        print(f"Predicted: {idx} ({name})  Confidence: {conf * 100:.2f}%")
        return idx, name, conf, probs
