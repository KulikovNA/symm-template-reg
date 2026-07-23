import unittest

from tests.boundary_augmentation_test_utils import run


class BoundaryDilationTest(unittest.TestCase):
    def test_dilation_adds_only_bounded_candidates(self):
        result = run("dilate")
        metadata = result["metadata"]
        self.assertGreater(metadata["added_total_count"], 0)
        self.assertLessEqual(metadata["added_fraction"], 0.20)
        self.assertEqual(metadata["removed_shell_count"], 0)


if __name__ == "__main__":
    unittest.main()
