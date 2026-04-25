"""Shared visualization utilities used by both TF and PyTorch Grad-CAM backends."""
from __future__ import annotations

import os

import cv2
import numpy as np


def load_image_bgr(img_path: str) -> np.ndarray:
    """Load a BGR image from disk with descriptive errors."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Image not found: '{img_path}'. Verify the path is correct."
        )
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(
            f"cv2.imread could not decode '{img_path}'. "
            "The file may be corrupt or an unsupported format."
        )
    return img


def build_overlay(
    heatmap: np.ndarray,
    img_bgr: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Resize a 2-D Grad-CAM heatmap to match *img_bgr* and blend them with JET colormap.

    Parameters
    ----------
    heatmap:
        2-D float array (H_cam, W_cam), values in [0, 1].
    img_bgr:
        BGR uint8 image (H, W, 3) — the original input image.
    alpha:
        Weight of the original image in the blend.
        0 = heatmap only, 1 = original image only.

    Returns
    -------
    np.ndarray
        BGR uint8 overlay with shape (H, W, 3).
    """
    if heatmap is None or heatmap.size == 0:
        raise ValueError("Heatmap is empty (None or zero elements).")
    if heatmap.ndim != 2:
        raise ValueError(
            f"Heatmap must be 2-D, got shape {heatmap.shape}. "
            "compute_gradcam() must return a 2-D array."
        )
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image is empty.")

    h, w = img_bgr.shape[:2]
    hm = heatmap.astype(np.float32)
    hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

    out = cv2.addWeighted(img_bgr, alpha, hm_color, 1.0 - alpha, 0)
    if out is None or out.size == 0:
        raise ValueError("cv2.addWeighted produced an empty result.")
    return out


def save_heatmap_image(
    output_img: np.ndarray,
    name: str,
    folder: str = "heatmap",
) -> str:
    """
    Write *output_img* to ``{folder}/{name}``.

    Appends ``.jpg`` if *name* has no file extension.

    Parameters
    ----------
    output_img:
        BGR uint8 image returned by :func:`build_overlay`.
    name:
        Filename (e.g. ``"result.jpg"`` or ``"result"``).
    folder:
        Output directory, created if it does not exist.

    Returns
    -------
    str
        Full path of the file written.
    """
    if output_img is None or (isinstance(output_img, np.ndarray) and output_img.size == 0):
        raise ValueError("save_heatmap_image received an empty image.")

    os.makedirs(folder, exist_ok=True)

    _, ext = os.path.splitext(name)
    if not ext.strip():
        name = name + ".jpg"

    path = os.path.join(folder, name)
    ok = cv2.imwrite(path, output_img)
    if not ok:
        raise IOError(
            f"cv2.imwrite failed for '{path}'. "
            "Check the directory is writable and the extension is supported."
        )

    print(f"Saved heatmap to: {path}")
    return path
