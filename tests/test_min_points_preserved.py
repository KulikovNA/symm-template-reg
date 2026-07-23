import unittest

from tests.boundary_augmentation_test_utils import run


class MinimumPointsTest(unittest.TestCase):
    def test_erosion_never_crosses_minimum(self):
        result = run(
            "erode",
            min_points_after_augmentation=24,
            max_removed_fraction=1.0,
        )
        self.assertGreaterEqual(len(result["points_C"]), 24)


if __name__ == "__main__":
    unittest.main()
