import unittest

from tests.boundary_augmentation_test_utils import run


class BoundaryMixedTest(unittest.TestCase):
    def test_mixed_erodes_and_dilates(self):
        metadata = run("mixed")["metadata"]
        self.assertGreater(metadata["removed_shell_count"], 0)
        self.assertGreater(metadata["added_total_count"], 0)


if __name__ == "__main__":
    unittest.main()
