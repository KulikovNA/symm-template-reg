import inspect
import unittest

from symm_template_reg.datasets.boundary_augmentation import (
    BoundaryMaskAugmentation,
)


class GTPoseLeakageTest(unittest.TestCase):
    def test_gt_pose_is_used_only_inside_target_construction(self):
        source = inspect.getsource(BoundaryMaskAugmentation.apply)
        self.assertIn("_transform_inverse", source)
        self.assertNotIn('"T_C_from_O":', source)


if __name__ == "__main__":
    unittest.main()
