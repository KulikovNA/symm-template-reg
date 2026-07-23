import unittest

from tests.boundary_augmentation_test_utils import run


class AlignedAugmentationTest(unittest.TestCase):
    def test_points_targets_uv_and_labels_remain_aligned(self):
        result = run("mixed")
        count = len(result["points_C"])
        self.assertEqual(len(result["target_points_O"]), count)
        self.assertEqual(len(result["pixel_uv"]), count)
        self.assertEqual(len(result["source_labels"]), count)


if __name__ == "__main__":
    unittest.main()
