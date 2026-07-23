import unittest

import numpy as np

from tests.boundary_augmentation_test_utils import run


class AddedTargetTest(unittest.TestCase):
    def test_added_targets_are_finite_and_on_template_plane(self):
        result = run("dilate")
        added = result["metadata"]["added_total_count"]
        self.assertGreater(added, 0)
        targets = result["target_points_O"][-added:]
        self.assertTrue(np.isfinite(targets).all())
        np.testing.assert_allclose(targets[:, 2], 0.0, atol=1e-6)
        self.assertLessEqual(
            result["metadata"]["max_template_projection_distance_mm"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
