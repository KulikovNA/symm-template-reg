from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.heads.residual_pose_hypothesis_head import (
    compose_camera_residual,
)


class ResidualPoseCompositionTest(unittest.TestCase):
    def test_zero_residual_equals_base(self) -> None:
        base = torch.eye(4).unsqueeze(0)
        base[:, :3, 3] = torch.tensor([[0.1, -0.2, 0.3]])
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).view(1, 1, 6)
        pose, residual = compose_camera_residual(
            base, identity_6d, torch.zeros(1, 1, 3), torch.tensor([0.1])
        )
        self.assertTrue(torch.allclose(pose[:, 0], base, atol=1e-6))
        self.assertTrue(torch.allclose(residual, torch.eye(4).view(1, 1, 4, 4), atol=1e-6))

    def test_translation_residual_is_camera_frame_scaled(self) -> None:
        base = torch.eye(4).unsqueeze(0)
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).view(1, 1, 6)
        pose, _ = compose_camera_residual(
            base,
            identity_6d,
            torch.tensor([[[1.0, 0.0, 0.0]]]),
            torch.tensor([0.02]),
        )
        self.assertTrue(torch.allclose(pose[0, 0, :3, 3], torch.tensor([0.02, 0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
