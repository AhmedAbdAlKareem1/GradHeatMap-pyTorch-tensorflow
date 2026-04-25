"""
Example: Grad-CAM on a PyTorch ResNet model (binary or multiclass).

Run:
    pip install "gradheatmap[torch]"
    python examples/torch_resnet.py --image path/to/image.jpg

Loading a trained checkpoint::

    python examples/torch_resnet.py \\
        --image cat.jpg \\
        --checkpoint resnet50_catdog_best.pth \\
        --classes cat dog
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torchvision import models

from gradheatmap import HeatMapPyTorch


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_resnet50(num_classes: int = 2, weights=None) -> nn.Module:
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes if num_classes > 2 else 1)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM demo — PyTorch ResNet50")
    parser.add_argument("--image",      required=True,              help="Path to input image")
    parser.add_argument("--checkpoint", default=None,               help="Path to .pth checkpoint")
    parser.add_argument("--classes",    nargs="+", default=["cat", "dog"], help="Class names")
    parser.add_argument("--output",     default="result_torch_resnet.jpg")
    parser.add_argument("--alpha",      type=float, default=0.4)
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224],
                        metavar=("H", "W"), help="Input image size")
    args = parser.parse_args()

    num_classes = len(args.classes)
    model = build_resnet50(num_classes=num_classes)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        class_names = ckpt.get("class_names", args.classes)
    else:
        print("No checkpoint given — using random weights for demo.")
        class_names = args.classes

    model.eval()

    hm = HeatMapPyTorch(
        model=model,
        img_path=args.image,
        class_names=class_names,
        preprocess="resnet50",                    # enables ImageNet normalization
        image_size=tuple(args.image_size),
    )

    overlay = hm.overlay_heatmap(alpha=args.alpha)
    hm.save_heat_img(args.output, overlay)
    print("Done.")


if __name__ == "__main__":
    main()
