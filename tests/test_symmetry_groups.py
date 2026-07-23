from __future__ import annotations

import math
import unittest

import torch

from symm_template_reg.models.pose.metrics import so2_pose_errors
from symm_template_reg.models.symmetry.groups import (
    CyclicGroup,
    SO2Group,
    group_angles,
    intersect_groups,
)
from symm_template_reg.models.symmetry.hypothesis_expander import (
    equivalent_gt_pose_set,
    equivalent_gt_poses,
    symmetry_transforms,
)


class TestSymmetryGroups(unittest.TestCase):
    def test_required_intersections(self) -> None:
        self.assertEqual(intersect_groups(CyclicGroup(4), CyclicGroup(2)), CyclicGroup(2))
        self.assertEqual(intersect_groups(CyclicGroup(4), SO2Group()), CyclicGroup(4))
        self.assertEqual(intersect_groups(CyclicGroup(12), CyclicGroup(8)), CyclicGroup(4))
        self.assertEqual(intersect_groups(CyclicGroup(7), CyclicGroup(5)), CyclicGroup(1))
        self.assertEqual(intersect_groups(SO2Group(), SO2Group()), SO2Group())

    def test_c4_produces_four_exact_equivalent_poses(self) -> None:
        gt = torch.eye(4, dtype=torch.float64)
        gt[:3, 3] = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
        poses = equivalent_gt_poses(gt, CyclicGroup(4), axis=[0.0, 1.0, 0.0])
        self.assertEqual(tuple(poses.shape), (4, 4, 4))
        self.assertTrue(torch.allclose(poses[0], gt))
        expected = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(poses[1, :3, :3], expected, atol=1e-12))
        determinants = torch.linalg.det(poses[:, :3, :3])
        self.assertTrue(torch.allclose(determinants, torch.ones_like(determinants)))

    def test_rotation_about_nonzero_axis_origin_fixes_origin(self) -> None:
        origin = torch.tensor([1.0, 2.0, 3.0])
        transforms = symmetry_transforms(CyclicGroup(4), [0.0, 1.0, 0.0], origin)
        homogeneous_origin = torch.cat((origin, torch.ones(1)))
        transformed = torch.matmul(transforms, homogeneous_origin)
        self.assertTrue(torch.allclose(transformed[:, :3], origin.expand(4, 3), atol=1e-6))

    def test_so2_requires_explicit_sampling_at_group_level(self) -> None:
        with self.assertRaisesRegex(ValueError, "continuous"):
            group_angles(SO2Group())
        angles = group_angles(SO2Group(), so2_num_samples=8, dtype=torch.float64)
        self.assertEqual(tuple(angles.shape), (8,))
        self.assertAlmostEqual(float(angles[-1]), 7.0 * math.pi / 4.0)

    def test_so2_pose_set_marks_finite_sampling_non_exhaustive(self) -> None:
        pose_set = equivalent_gt_pose_set(
            torch.eye(4),
            SO2Group(),
            axis=[0.0, 1.0, 0.0],
            so2_num_samples=12,
        )
        self.assertEqual(tuple(pose_set.poses.shape), (12, 4, 4))
        self.assertTrue(pose_set.is_continuous)
        self.assertFalse(pose_set.exhaustive)
        self.assertEqual(pose_set.metadata_dict()["effective_group"], {"type": "SO2"})

    def test_so2_metric_ignores_twist(self) -> None:
        target = torch.eye(4)
        twist = symmetry_transforms(
            SO2Group(),
            [0.0, 1.0, 0.0],
            so2_num_samples=4,
        )[1]
        errors = so2_pose_errors(twist, target, axis_O=[0.0, 1.0, 0.0])
        self.assertAlmostEqual(float(errors["axis_error_rad"]), 0.0, places=6)
        self.assertAlmostEqual(float(errors["translation_m"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
