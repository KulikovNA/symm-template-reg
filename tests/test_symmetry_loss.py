import unittest

import torch

from symm_template_reg.models.losses import SymmetryPoseLoss
from symm_template_reg.models.symmetry import (
    CyclicGroup,
    SO2Group,
    equivalent_gt_poses,
    symmetry_transforms,
)


class SymmetryLossTest(unittest.TestCase):
    def test_c4_equivalent_target_has_minimum_loss(self):
        target = torch.eye(4)
        equivalents = equivalent_gt_poses(target, CyclicGroup(4), axis=[0.0, 1.0, 0.0])
        prediction = equivalents[2]
        loss = SymmetryPoseLoss()(prediction, target, equivalents)
        self.assertLess(float(loss), 1e-3)

    def test_so2_twist_about_nonzero_origin_is_unpenalized(self):
        target = torch.eye(4)
        origin = torch.tensor([0.3, -0.2, 0.1])
        prediction = symmetry_transforms(
            SO2Group(), [0.0, 1.0, 0.0], origin, so2_num_samples=4
        )[1]
        loss = SymmetryPoseLoss()(
            prediction,
            target,
            continuous_axis_O=torch.tensor([0.0, 1.0, 0.0]),
            continuous_origin_O=origin,
        )
        self.assertLess(float(loss), 1e-3)


if __name__ == "__main__":
    unittest.main()
