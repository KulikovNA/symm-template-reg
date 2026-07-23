import unittest

import numpy as np

from symm_template_reg.datasets.boundary_augmentation import (
    BoundaryMaskAugmentation,
)
from tests.boundary_augmentation_test_utils import case


class GTPoseMutationTest(unittest.TestCase):
    def test_pose_input_is_not_mutated(self):
        payload = case()
        before = payload["T_C_from_O"].copy()
        BoundaryMaskAugmentation(
            {
                "enabled": True,
                "apply_probability": 1.0,
                "mode": "mixed",
                "min_points_after_augmentation": 5,
            }
        ).apply(**payload)
        np.testing.assert_array_equal(payload["T_C_from_O"], before)


if __name__ == "__main__":
    unittest.main()
