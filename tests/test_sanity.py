import unittest
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestSanity(unittest.TestCase):
    def test_dependencies(self):
        """Check if critical libraries are importable."""
        try:
            import cv2
            import rembg
            import pycolmap
            import open3d
        except ImportError as e:
            self.fail(f"Dependency missing: {e}")

    def test_preprocess_logic(self):
        """Check variance of laplacian logic."""
        from preprocess import variance_of_laplacian
        # Create a dummy image (black square)
        img = np.zeros((100, 100), dtype=np.uint8)
        score = variance_of_laplacian(img)
        self.assertIsInstance(score, float)
        self.assertEqual(score, 0.0)

        # Noise image
        img_noise = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        score_noise = variance_of_laplacian(img_noise)
        self.assertGreater(score_noise, 0.0)

    def test_reconstruct_imports(self):
        """Check if pycolmap exposes necessary options."""
        import pycolmap
        self.assertTrue(hasattr(pycolmap, "ImageReaderOptions"), "pycolmap should have ImageReaderOptions")
        self.assertTrue(hasattr(pycolmap, "extract_features"), "pycolmap should have extract_features")

if __name__ == '__main__':
    unittest.main()
