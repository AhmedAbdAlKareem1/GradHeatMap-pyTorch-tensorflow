"""Shared pytest fixtures for GradHeatMap tests."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def dummy_image_path(tmp_path):
    """Write a 64x64 BGR image to a temp file and return its path."""
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = [180, 90, 40]   # a colored square
    path = str(tmp_path / "test_img.jpg")
    cv2.imwrite(path, img)
    return path
