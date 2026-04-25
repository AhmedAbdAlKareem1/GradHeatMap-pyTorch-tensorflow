"""
Example: Grad-CAM on a TensorFlow transfer-learning model (MobileNetV2 backbone).

Run:
    pip install "gradheatmap[tf]"
    python examples/tf_transfer_learning.py --image path/to/image.jpg --classes cat dog
"""
from __future__ import annotations

import argparse

import tensorflow as tf

from gradheatmap import HeatMap


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_transfer_model(num_classes: int = 2, image_size: int = 224):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inp = tf.keras.Input((image_size, image_size, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    if num_classes == 2:
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    else:
        out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="mobilenetv2_transfer")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM demo — TF transfer learning")
    parser.add_argument("--image",   required=True,            help="Path to input image")
    parser.add_argument("--model",   default=None,             help="Path to .keras/.h5 (optional)")
    parser.add_argument("--classes", nargs="+", default=["cat", "dog"], help="Class names")
    parser.add_argument("--output",  default="result_tf_transfer.jpg")
    parser.add_argument("--alpha",   type=float, default=0.4)
    args = parser.parse_args()

    if args.model:
        model = tf.keras.models.load_model(args.model, compile=False)
    else:
        print("No model path given — building demo MobileNetV2 transfer model.")
        model = build_transfer_model(num_classes=len(args.classes))

    hm = HeatMap(
        model=model,
        img_path=args.image,
        class_names=args.classes,
        # preprocess is auto-detected from the MobileNetV2 backbone
    )

    overlay = hm.overlay_heatmap(alpha=args.alpha)
    hm.save_heat_img(args.output, overlay)
    print("Done.")


if __name__ == "__main__":
    main()
