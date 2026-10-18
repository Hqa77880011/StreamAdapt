import unittest
import numpy as np
from finite_data import render


class FiniteInputTests(unittest.TestCase):
    def test_geometry_changes_pixels_without_label_feature(self):
        rng = np.random.default_rng(9)
        circle = render("circle", [.5, .5], [200, 180, 140], 1, rng)
        square = render("square", [.5, .5], [200, 180, 140], 1, np.random.default_rng(9))
        self.assertEqual(circle.shape, (128, 128, 3))
        self.assertEqual(circle.dtype, np.uint8)
        self.assertFalse(np.array_equal(circle, square))
        self.assertEqual(int(circle[0, 0].mean()), int(square[0, 0].mean()))
