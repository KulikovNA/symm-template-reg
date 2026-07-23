from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.pose.pose_representation import (
    invert_transform,
    make_transform,
    transform_points,
    validate_transform,
)
from symm_template_reg.models.pose.rotation import (
    axis_angle_to_matrix,
    is_rotation_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


class TestRotation6D(unittest.TestCase):
    def test_batched_rotation_has_positive_unit_determinant(self) -> None:
        torch.manual_seed(7)
        values = torch.randn(2, 5, 6, dtype=torch.float64)
        rotation = rotation_6d_to_matrix(values)
        self.assertEqual(tuple(rotation.shape), (2, 5, 3, 3))
        identity = torch.eye(3, dtype=rotation.dtype).expand(2, 5, 3, 3)
        self.assertTrue(
            torch.allclose(rotation @ rotation.transpose(-1, -2), identity, atol=1e-10)
        )
        self.assertTrue(
            torch.allclose(torch.linalg.det(rotation), torch.ones(2, 5, dtype=torch.float64))
        )

    def test_matrix_6d_round_trip(self) -> None:
        axis_angles = torch.tensor(
            [[0.2, -0.1, 0.3], [0.0, torch.pi / 2.0, 0.0]], dtype=torch.float64
        )
        rotation = axis_angle_to_matrix(axis_angles)
        reconstructed = rotation_6d_to_matrix(matrix_to_rotation_6d(rotation))
        self.assertTrue(torch.allclose(rotation, reconstructed, atol=1e-10))

    def test_degenerate_inputs_still_map_to_so3(self) -> None:
        values = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]]
        )
        rotation = rotation_6d_to_matrix(values)
        self.assertTrue(bool(is_rotation_matrix(rotation).all()))
        self.assertTrue(torch.allclose(rotation[0], torch.eye(3)))

    def test_backward_is_finite(self) -> None:
        torch.manual_seed(11)
        values = torch.randn(16, 6, requires_grad=True)
        rotation = rotation_6d_to_matrix(values)
        loss = (rotation * torch.arange(9, dtype=rotation.dtype).reshape(3, 3)).sum()
        loss.backward()
        self.assertIsNotNone(values.grad)
        assert values.grad is not None
        self.assertTrue(bool(torch.isfinite(values.grad).all()))

    def test_transform_round_trip(self) -> None:
        rotation = axis_angle_to_matrix(torch.tensor([0.1, 0.2, -0.3]))
        transform = make_transform(rotation, torch.tensor([0.4, -0.2, 1.0]))
        points = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.5, -0.1]])
        restored = transform_points(invert_transform(transform), transform_points(transform, points))
        self.assertTrue(torch.allclose(restored, points, atol=1e-6))
        self.assertTrue(bool(validate_transform(transform)))


if __name__ == "__main__":
    unittest.main()
