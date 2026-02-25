GradHeatmap
GradHeatmap is a Python library designed to simplify the generation of Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps. It automates the process of backbone detection and layer identification for both TensorFlow/Keras and PyTorch models, removing the need for manual architecture inspection or hardcoded layer names.



Features
Framework Agnostic: Unified support for TensorFlow/Keras (.keras, .h5) and PyTorch models.

Automatic Detection: Automatically identifies backbones (ResNet, VGG, MobileNet, EfficientNet, etc.) and the final convolutional layer.

Classification Support: Works with binary (sigmoid/single logit) and multi-class (softmax/multi-logit) outputs.

Smart Preprocessing: Detects internal tf.keras.layers.Rescaling in TensorFlow and provides ImageNet normalization for PyTorch.

Transfer Learning Ready: Seamlessly handles custom CNNs and models built via transfer learning.

Visualization: OpenCV JET heatmap overlay with adjustable alpha blending.

Validation: Built-in safe output validation and fallback normalization.

Installation
Install the package directly from GitHub:

```
pip install git+https://github.com/AhmedAbdAlKareem1/gradheatmap.git
```
Optional Extras
To install dependencies for specific frameworks:
# For TensorFlow
```
pip install "gradheatmap[tf]"
```
# For PyTorch
```
pip install "gradheatmap[torch]"
```
Quick Start (TensorFlow / Keras)
```
from gradheatmap import HeatMap

model_path = "your_model.keras"   # Supports .keras or .h5
image_path = "your_image.jpg"
class_names = ["class0", "class1"]

heat = HeatMap(
    model=model_path,
    img_path=image_path,
    class_names=class_names
)

overlay = heat.overlay_heatmap(alpha=0.4)
heat.save_heat_img("result.jpg", overlay)
```
Project Structure Output:

Plaintext
heatmap/
└── result.jpg
<p align="center">
<img src="heatmap1.jpg" width="300">
<img src="heatmap2.jpg" width="300">
<img src="heatmap3.jpg" width="300">
</p>
Quick Start (PyTorch)
```
import torch
import torch.nn as nn
from torchvision import models
from gradheatmap import HeatMapPyTorch

def build_resnet50(num_classes=2):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# Load model and weights
ckpt = torch.load("resnet50_catdog_best.pth", map_location="cpu")
model = build_resnet50(num_classes=2)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

heat = HeatMapPyTorch(
    model=model,
    img_path="image.jpg",
    class_names=ckpt.get("class_names", ["Cat", "Dog"]),
    preprocess="resnet50",
    image_size=(224, 224)
)

overlay = heat.overlay_heatmap(alpha=0.4)
heat.save_heat_img("result_torch.jpg", overlay)
```
```
Terminal Output Example
When running the generator, the library provides automated feedback on the model architecture:

Plaintext
Detected Model : vgg16
Detected Image Size = (224, 224)
Detected Backbone = vgg16
Last Convo Layer : block5_conv3
Class: 0 class0  Confidence: 100.00%
Successfully saved heatmap to: heatmap/result.jpg
```
Requirements
Python >= 3.8

OpenCV

NumPy

TensorFlow >= 2.x (Optional)

PyTorch >= 1.x (Optional)

License
This project is licensed under the MIT License.






