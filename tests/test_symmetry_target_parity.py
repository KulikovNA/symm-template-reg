from __future__ import annotations

import unittest
from pathlib import Path

import torch

from symm_template_reg.datasets import FragmentTemplateRegistrationDataset
from symm_template_reg.models.losses import SymmetryPoseLoss
from symm_template_reg.models.pose.pose_representation import transform_points
from symm_template_reg.models.symmetry.groups import (
    CyclicGroup,
    SO2Group,
    intersect_rotation_groups,
)
from symm_template_reg.models.symmetry.hypothesis_expander import (
    equivalent_gt_pose_set,
    symmetry_transforms,
    visualization_equivalent_pose_set,
)
from symm_template_reg.models.symmetry.metadata import load_symmetry_metadata
from symm_template_reg.models.symmetry.targets import build_symmetry_targets
from symm_template_reg.visualization.symmetry_debug import (
    camera_points_in_hypothesis_object,
)


DATASET_ROOT = Path(
    "/home/nikita/data_generator/generation_dataset/generation_synthetic/output/"
    "fragment_template_registration/differBig/2026-07-08/test"
)
SIDECAR = DATASET_ROOT / "models" / "object_000004__scale_0p05.symmetry.json"


class SymmetryTargetMathTest(unittest.TestCase):
    def test_required_groups_and_exact_cyclic_hypothesis_counts(self) -> None:
        self.assertEqual(intersect_rotation_groups(SO2Group(), CyclicGroup(10)), CyclicGroup(10))
        self.assertEqual(intersect_rotation_groups(CyclicGroup(10), CyclicGroup(4)), CyclicGroup(2))
        self.assertEqual(intersect_rotation_groups(SO2Group(), CyclicGroup(4)), CyclicGroup(4))
        identity = torch.eye(4, dtype=torch.float64)
        c10 = equivalent_gt_pose_set(identity, CyclicGroup(10), axis=[0, 1, 0])
        c4 = equivalent_gt_pose_set(identity, CyclicGroup(4), axis=[0, 1, 0])
        self.assertEqual(c10.num_hypotheses, 10)
        self.assertEqual(c4.num_hypotheses, 4)
        for poses in (c10.poses, c4.poses):
            determinants = torch.linalg.det(poses[:, :3, :3])
            torch.testing.assert_close(determinants, torch.ones_like(determinants))

    def test_hypotheses_fix_configured_axis_and_origin(self) -> None:
        axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        origin = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64)
        transforms = symmetry_transforms(CyclicGroup(10), axis, origin, dtype=torch.float64)
        rotated_axis = transforms[:, :3, :3] @ axis
        transformed_origin = transform_points(transforms, origin.reshape(1, 3)).squeeze(-2)
        torch.testing.assert_close(rotated_axis, axis.expand_as(rotated_axis), atol=1e-12, rtol=0)
        torch.testing.assert_close(
            transformed_origin, origin.expand_as(transformed_origin), atol=1e-12, rtol=0
        )

    def test_pose_convention_round_trip_uses_inverse_generated_hypothesis(self) -> None:
        gt = torch.eye(4)
        gt[:3, 3] = torch.tensor([0.1, -0.2, 0.7])
        points_O = torch.tensor([[0.01, -0.02, 0.03], [-0.03, 0.04, 0.02]])
        points_C = transform_points(gt, points_O)
        poses = equivalent_gt_pose_set(gt, CyclicGroup(4), axis=[0, 1, 0]).poses
        for pose in poses:
            recovered = camera_points_in_hypothesis_object(points_C, pose)
            reconstructed = transform_points(pose, recovered)
            torch.testing.assert_close(reconstructed, points_C, atol=1e-6, rtol=1e-6)


@unittest.skipUnless(SIDECAR.is_file(), "real symmetry fixture unavailable")
class RealDatasetSymmetryTargetParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = FragmentTemplateRegistrationDataset(
            DATASET_ROOT,
            observed_policy="farthest_point_up_to_max",
            min_observed_points=128,
            max_observed_points=4096,
            template_fine_points=2048,
            template_coarse_points=512,
        )

    def test_debug_builder_and_dataset_targets_are_identical(self) -> None:
        sample = self.dataset[0]
        targets = build_symmetry_targets(
            sample["gt"]["points_O_corresponding"],
            sample["gt"]["T_C_from_O"],
            sample["template"]["symmetry_metadata"],
        )
        self.assertTrue(
            torch.equal(targets.active_regions, sample["gt"]["active_symmetry_regions"])
        )
        self.assertEqual(
            targets.effective_group.to_dict(), sample["gt"]["effective_symmetry_group"]
        )
        torch.testing.assert_close(
            targets.equivalent_poses, sample["gt"]["equivalent_T_C_from_O"]
        )
        self.assertEqual(targets.effective_group, CyclicGroup(2))
        self.assertEqual(targets.equivalent_pose_set.num_hypotheses, 2)
        loss = SymmetryPoseLoss()(
            targets.equivalent_poses[0],
            sample["gt"]["T_C_from_O"],
            symmetry_targets=targets,
        )
        self.assertTrue(torch.isfinite(loss))

    def test_so2_is_analytic_and_gallery_is_explicitly_finite(self) -> None:
        metadata = load_symmetry_metadata(SIDECAR)
        assert metadata is not None
        points = torch.tensor([[0.0, -0.04, 0.0], [0.01, -0.045, 0.01]])
        targets = build_symmetry_targets(points, torch.eye(4), metadata)
        self.assertEqual(targets.effective_group, SO2Group())
        self.assertEqual(targets.training_target_type, "continuous_analytic")
        self.assertFalse(targets.equivalent_pose_set.exhaustive)
        visualization = visualization_equivalent_pose_set(
            torch.eye(4),
            metadata,
            effective_group=targets.effective_group,
            so2_visualization_samples=12,
        )
        self.assertEqual(visualization.num_hypotheses, 12)
        self.assertFalse(visualization.exhaustive)


if __name__ == "__main__":
    unittest.main()
