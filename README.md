# GradHeatMap

Automatic **Grad-CAM** heatmap generation for **TensorFlow/Keras** and **PyTorch** classification models.

No manual layer names or architecture inspection needed — the library auto-detects the last convolutional layer, the preprocessing function, and the classification head type.

---

## Features

- **Framework agnostic** — unified API for TensorFlow/Keras (`.keras`, `.h5`) and PyTorch
- **Automatic backbone detection** — ResNet, VGG, MobileNet, EfficientNet, DenseNet, and more
- **Automatic last-conv detection** — walks nested sub-models recursively
- **Smart preprocessing** — detects `tf.keras.layers.Rescaling`, backbone-specific normalization, or falls back to `[0, 1]` scaling
- **Binary & multiclass** — sigmoid single-output, 2-class softmax, multiclass softmax
- **Explicit target class** — optionally force Grad-CAM to explain a specific class
- **OpenCV JET overlay** — adjustable alpha blending
- **Clean errors** — descriptive messages for missing images, bad paths, and unsupported shapes

---

## Installation

```bash
# Core (no framework — import-only)
pip install gradheatmap

# With TensorFlow support
pip install "gradheatmap[tf]"

# With PyTorch support
pip install "gradheatmap[torch]"

# Development (includes pytest, build tools)
pip install -e ".[dev]"
```

---

## Quick Start — TensorFlow / Keras

### Custom CNN (binary sigmoid)

```python
from gradheatmap import HeatMap

hm = HeatMap(
    model="my_model.keras",       # or a tf.keras.Model instance
    img_path="cat.jpg",
    class_names=["cat", "dog"],
)
overlay = hm.overlay_heatmap(alpha=0.4)
hm.save_heat_img("result.jpg", overlay)
```

### Transfer learning (auto-detects backbone preprocessing)

```python
import tensorflow as tf
from gradheatmap import HeatMap

base = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False

inp = tf.keras.Input((224, 224, 3))
x = base(inp, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inp, x)

hm = HeatMap(model=model, img_path="cat.jpg", class_names=["cat", "dog"])
# VGG16 preprocessing is detected automatically from the nested backbone
hm.save_heat_img("result_vgg.jpg", hm.overlay_heatmap())
```

### Multiclass

```python
from gradheatmap import HeatMap

hm = HeatMap(
    model="flowers_model.keras",
    img_path="rose.jpg",
    class_names=["daisy", "dandelion", "rose", "sunflower", "tulip"],
    target_class=2,   # force explain "rose" class
)
hm.save_heat_img("result_flower.jpg", hm.overlay_heatmap())
```

---

## Quick Start — PyTorch

### Custom CNN

```python
import torch.nn as nn
from gradheatmap import HeatMapPyTorch

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Linear(64, 1), nn.Sigmoid(),
)

hm = HeatMapPyTorch(
    model=model,
    img_path="cat.jpg",
    class_names=["cat", "dog"],
    image_size=(224, 224),
)
hm.save_heat_img("result.jpg", hm.overlay_heatmap())
```

### ResNet50 with checkpoint

```python
import torch
import torch.nn as nn
from torchvision import models
from gradheatmap import HeatMapPyTorch

def build_resnet50(num_classes=2):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

ckpt = torch.load("resnet50_catdog_best.pth", map_location="cpu")
model = build_resnet50(num_classes=2)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

hm = HeatMapPyTorch(
    model=model,
    img_path="cat.jpg",
    class_names=ckpt.get("class_names", ["cat", "dog"]),
    preprocess="resnet50",      # enables ImageNet mean/std normalization
    image_size=(224, 224),
)
hm.save_heat_img("result_resnet.jpg", hm.overlay_heatmap())
```

---

## Output

All heatmaps are written to the `heatmap/` directory by default (created automatically):

```
heatmap/
└── result.jpg
```

---

## API Reference

### `HeatMap` (TensorFlow/Keras)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `tf.keras.Model` or `str` | required | Model instance or path to `.keras`/`.h5` |
| `img_path` | `str` | required | Path to input image |
| `class_names` | `list[str]` | required | Human-readable class labels |
| `target_class` | `int` or `None` | `None` | Class index to explain; `None` = predicted class |
| `preprocess` | callable, str, or `None` | `None` | Override preprocessing (auto-detected if `None`) |

### `HeatMapPyTorch`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | required | Trained model (loaded, set to eval mode) |
| `img_path` | `str` | required | Path to input image |
| `class_names` | `list[str]` or `None` | `None` | Human-readable class labels |
| `target_class` | `int` or `None` | `None` | Class index to explain; `None` = predicted class |
| `preprocess` | callable, str, or `None` | `None` | `"resnet50"`, `"vgg16"`, custom fn, or `None` |
| `image_size` | `(H, W)` | `(224, 224)` | Resize target |
| `device` | `torch.device` or `None` | `None` | Auto-detect CUDA/CPU if `None` |

#### Methods (both classes)

| Method | Returns | Description |
|---|---|---|
| `compute_gradcam()` | `np.ndarray (H, W)` | Raw 2-D heatmap, values in `[0, 1]` |
| `overlay_heatmap(alpha=0.4)` | `np.ndarray (H, W, 3)` | BGR image with JET heatmap blended in |
| `save_heat_img(name, img, folder="heatmap")` | `str` | Write overlay to disk; returns the path |

---

## Supported backbone preprocessing (auto-detected)

`vgg16`, `vgg19`, `resnet`, `resnet50`, `resnetv2`, `mobilenet`, `mobilenetv2`, `mobilenetv3`,
`efficientnet`, `efficientnetv2`, `densenet`, `inception`, `inceptionresnet`,
`xception`, `nasnet`, `convnext`, `regnet`

---

## Terminal output example

```
Detected backbone     : vgg16
Detected image size   : (224, 224)
Last conv layer       : block5_conv3
Auto-detected backbone preprocessing: 'vgg16'.
Predicted: 0 (cat)  Confidence: 98.34%
Saved heatmap to: heatmap/result.jpg
```

---

## Running tests

```bash
pip install -e ".[tf,torch,dev]"
python -m pytest
```

---

## Requirements

- Python ≥ 3.8
- OpenCV ≥ 4.7
- NumPy

Optional:
- TensorFlow ≥ 2.10 (`pip install "gradheatmap[tf]"`)
- PyTorch ≥ 1.10 + torchvision (`pip install "gradheatmap[torch]"`)

---

## License

MIT
