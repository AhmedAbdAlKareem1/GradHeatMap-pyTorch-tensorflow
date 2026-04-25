"""
Example: Grad-CAM on a custom TensorFlow CNN (binary classification).

Run:
    pip install "gradheatmap[tf]"
    python examples/tf_custom_cnn.py --image path/to/image.jpg
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from gradheatmap import HeatMap


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_model(input_shape=(224, 224, 3)):
    inp = tf.keras.Input(input_shape)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="custom_binary_cnn")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM demo — custom TF CNN")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--model",  default=None,  help="Path to .keras/.h5 file (optional)")
    parser.add_argument("--output", default="result_tf_cnn.jpg", help="Output filename")
    parser.add_argument("--alpha",  type=float, default=0.4,  help="Heatmap alpha blend")
    args = parser.parse_args()

    if args.model:
        print(f"Loading model from: {args.model}")
        model = tf.keras.models.load_model(args.model, compile=False)
    else:
        print("No model path given — building a random-weight model for demo.")
        model = build_model()

    hm = HeatMap(
        model=model,
        img_path=args.image,
        class_names=["negative", "positive"],
    )

    overlay = hm.overlay_heatmap(alpha=args.alpha)
    hm.save_heat_img(args.output, overlay)
    print("Done.")


if __name__ == "__main__":
    main()
