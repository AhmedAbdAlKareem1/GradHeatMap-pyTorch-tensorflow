import unittest
import os
import sys

# Ensure we import the local checkout, not an installed package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestBaseImport(unittest.TestCase):
    def test_import_gradheatmap_base(self):
        import gradheatmap  # noqa: F401

    def test_instantiating_backend_without_dep_raises(self):
        import gradheatmap

        try:
            import tensorflow  # noqa: F401
            tf_available = True
        except Exception:
            tf_available = False

        if not tf_available:
            with self.assertRaises(ImportError):
                gradheatmap.HeatMap(model=None, img_path="x.jpg", class_names=[])  # type: ignore[arg-type]

        try:
            import torch  # noqa: F401
            torch_available = True
        except Exception:
            torch_available = False

        if not torch_available:
            with self.assertRaises(ImportError):
                gradheatmap.HeatMapPyTorch(model=None, img_path="x.jpg", class_names=[])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

