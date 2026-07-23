import unittest

import numpy as np

from tests.boundary_augmentation_test_utils import run


class DeterministicSeedTest(unittest.TestCase):
    def test_same_seed_is_exactly_repeatable(self):
        first = run("mixed")
        second = run("mixed")
        np.testing.assert_array_equal(first["points_C"], second["points_C"])
        np.testing.assert_array_equal(
            first["target_points_O"], second["target_points_O"]
        )
        self.assertEqual(first["metadata"], second["metadata"])


if __name__ == "__main__":
    unittest.main()
