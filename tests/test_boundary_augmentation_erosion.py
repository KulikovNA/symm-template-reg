import unittest

from tests.boundary_augmentation_test_utils import run


class BoundaryErosionTest(unittest.TestCase):
    def test_erosion_removes_only_bounded_boundary_rows(self):
        result = run("erode")
        metadata = result["metadata"]
        self.assertGreater(metadata["removed_shell_count"], 0)
        self.assertLessEqual(metadata["removed_fraction"], 0.20)
        self.assertEqual(metadata["added_total_count"], 0)


if __name__ == "__main__":
    unittest.main()
