from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.losses import SymmetryPoseLoss
from symm_template_reg.models.symmetry.groups import CyclicGroup, SO2Group
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms
from symm_template_reg.models.symmetry.targets import build_fragment_symmetry_targets
from tests.test_fragment_symmetry_targets import metadata, triangle_at_y


class SymmetryPoseOracleTest(unittest.TestCase):
    def test_c4_and_c10_equivalent_poses_have_near_zero_loss(self) -> None:
        sidecar = metadata()
        gt = torch.eye(4)
        for group in (CyclicGroup(4), CyclicGroup(10)):
            transforms = symmetry_transforms(
                group, sidecar.axis.direction, sidecar.axis.origin
            )
            loss = SymmetryPoseLoss()(
                gt @ transforms[-1], gt, equivalent_gt_poses=gt @ transforms
            )
            self.assertLess(float(loss), 1e-3)

    def test_so2_oracle_ignores_twist(self) -> None:
        points, faces = triangle_at_y(-0.043)
        targets = build_fragment_symmetry_targets(
            points, metadata(), fragment_faces=faces
        )
        self.assertEqual(targets.effective_group, SO2Group())
        twist = symmetry_transforms(
            CyclicGroup(7), metadata().axis.direction, metadata().axis.origin
        )[3]
        loss = SymmetryPoseLoss()(
            twist, torch.eye(4), symmetry_targets=targets
        )
        self.assertLess(float(loss), 1e-3)


if __name__ == "__main__":
    unittest.main()
