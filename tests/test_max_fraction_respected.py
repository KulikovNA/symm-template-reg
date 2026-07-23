import unittest

from tests.boundary_augmentation_test_utils import run


class FractionLimitTest(unittest.TestCase):
    def test_both_limits_are_respected(self):
        metadata = run(
            "mixed",
            max_removed_fraction=0.08,
            max_added_fraction=0.05,
        )["metadata"]
        self.assertLessEqual(metadata["removed_fraction"], 0.08)
        self.assertLessEqual(metadata["added_fraction"], 0.05)


if __name__ == "__main__":
    unittest.main()
