"""
Example: Grad-CAM for binary classification — both TF and PyTorch patterns.

Demonstrates:
- Sigmoid single-output (most common binary setup)
- Two-class softmax
- Choosing target_class explicitly
"""
from __future__ import annotations

# ── TensorFlow binary sigmoid ────────────────────────────────────────────────
TF_SIGMOID_EXAMPLE = """
import tensorflow as tf
from gradheatmap import HeatMap

# --- Model with sigmoid output ---
model = tf.keras.Sequential([
    tf.keras.Input((224, 224, 3)),
    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation="sigmoid"),   # sigmoid → P(positive)
])

hm = HeatMap(
    model=model,
    img_path="cat.jpg",
    class_names=["cat", "dog"],
    # target_class=None → explain the predicted class
    # target_class=0    → explain "cat" regardless of prediction
    # target_class=1    → explain "dog" regardless of prediction
)
overlay = hm.overlay_heatmap(alpha=0.4)
hm.save_heat_img("binary_result.jpg", overlay)
"""

# ── TensorFlow binary softmax ────────────────────────────────────────────────
TF_SOFTMAX_EXAMPLE = """
import tensorflow as tf
from gradheatmap import HeatMap

# --- Model with 2-class softmax output ---
model = tf.keras.Sequential([
    tf.keras.Input((224, 224, 3)),
    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation="softmax"),   # 2-class softmax
])

hm = HeatMap(model=model, img_path="cat.jpg", class_names=["cat", "dog"])
overlay = hm.overlay_heatmap()
hm.save_heat_img("binary_softmax_result.jpg", overlay)
"""

# ── PyTorch binary sigmoid ────────────────────────────────────────────────────
TORCH_SIGMOID_EXAMPLE = """
import torch.nn as nn
from gradheatmap import HeatMapPyTorch

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 1),
    nn.Sigmoid(),           # sigmoid → P(positive)
)

hm = HeatMapPyTorch(
    model=model,
    img_path="cat.jpg",
    class_names=["cat", "dog"],
    image_size=(224, 224),
)
overlay = hm.overlay_heatmap(alpha=0.4)
hm.save_heat_img("torch_binary_result.jpg", overlay)
"""

# ── PyTorch binary softmax ───────────────────────────────────────────────────
TORCH_SOFTMAX_EXAMPLE = """
import torch.nn as nn
from gradheatmap import HeatMapPyTorch

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 2),
    nn.Softmax(dim=1),      # 2-class softmax
)

hm = HeatMapPyTorch(
    model=model,
    img_path="cat.jpg",
    class_names=["cat", "dog"],
    image_size=(224, 224),
    target_class=1,         # always explain "dog" class
)
overlay = hm.overlay_heatmap()
hm.save_heat_img("torch_binary_softmax_result.jpg", overlay)
"""

if __name__ == "__main__":
    print("Binary classification examples (see source code for code snippets).")
    print(TF_SIGMOID_EXAMPLE)
    print(TORCH_SIGMOID_EXAMPLE)
