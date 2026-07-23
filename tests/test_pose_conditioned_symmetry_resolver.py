from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.losses import PoseSetLoss
from symm_template_reg.models.symmetry.groups import CyclicGroup
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms
from symm_template_reg.models.symmetry.pose_conditioned_resolver import (
    PoseConditionedSymmetryResolver,
)
from symm_template_reg.models.symmetry.targets import build_fragment_symmetry_targets
from tests.test_fragment_symmetry_targets import metadata


class PoseConditionedResolverTest(unittest.TestCase):
    def test_gt_pose_reproduces_gt_regions_and_c10_c4_intersection(self) -> None:
        points_O = torch.tensor(
            [[0.0, -0.020, 0.0], [0.01, -0.015, 0.0], [0.0, 0.030, 0.0]]
        )
        pose = torch.eye(4)
        pose[:3, 3] = torch.tensor([0.1, -0.2, 0.5])
        points_C = points_O @ pose[:3, :3].T + pose[:3, 3]
        target = build_fragment_symmetry_targets(
            points_O, metadata(), base_pose=pose
        )
        result = PoseConditionedSymmetryResolver().resolve(
            points_C[None],
            torch.ones((1, len(points_C)), dtype=torch.bool),
            pose[None, None],
            [metadata()],
            dict(min_points=1, min_fraction=0.0, boundary_tolerance_m=1e-6),
        )
        self.assertEqual(
            result.active_regions_per_pose[0][0].tolist(),
            target.active_regions.tolist(),
        )
        self.assertEqual(result.effective_group_per_pose[0][0], CyclicGroup(2))
        self.assertEqual(len(result.expanded_poses_per_base_pose[0][0]), 2)
        self.assertFalse(result.unresolved_flags[0][0])

    def test_unresolved_does_not_silently_fallback_to_learned_group(self) -> None:
        result = PoseConditionedSymmetryResolver().resolve(
            torch.tensor([[[0.0, 10.0, 0.0]]]),
            torch.ones((1, 1), dtype=torch.bool),
            torch.eye(4)[None, None],
            [metadata()],
            dict(unresolved_group_policy="base_pose_only"),
        )
        self.assertTrue(result.unresolved_flags[0][0])
        self.assertIsNone(result.effective_group_per_pose[0][0])
        self.assertEqual(len(result.expanded_poses_per_base_pose[0][0]), 1)

    def test_training_pose_loss_uses_gt_group_not_a_predicted_group(self) -> None:
        sidecar = metadata()
        gt = torch.eye(4)
        quarter_turn = symmetry_transforms(
            CyclicGroup(4), sidecar.axis.direction, sidecar.axis.origin
        )[1]
        predictions = quarter_turn[None, None]
        logits = torch.zeros((1, 1))
        criterion = PoseSetLoss(classification_weight=0.0)
        gt_group_loss = criterion(
            predictions,
            logits,
            gt[None],
            symmetry_metadata=[sidecar],
            effective_symmetry_groups=[{"type": "C", "order": 4}],
        )["loss_pose_set"]
        hypothetical_predicted_c1_loss = criterion(
            predictions,
            logits,
            gt[None],
            symmetry_metadata=[sidecar],
            effective_symmetry_groups=[{"type": "C", "order": 1}],
        )["loss_pose_set"]
        self.assertLess(float(gt_group_loss), float(hypothetical_predicted_c1_loss))


if __name__ == "__main__":
    unittest.main()
