from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.heads.conditioned_base_pose_head import ConditionedBasePoseHead
from symm_template_reg.models.pose.rotation import axis_angle_to_matrix, rotation_geodesic_distance


class ResidualCannotReplaceBasePoseTest(unittest.TestCase):
    def test_final_pose_remains_in_bounded_neighborhood_of_correspondence_pose(self) -> None:
        head = ConditionedBasePoseHead(embed_dim=8, hidden_dim=16, split_rotation_translation=True, output_mode="bounded_correction", max_rotation_correction_deg=5.0, max_translation_correction_m=0.003)
        reference = torch.eye(4)[None]
        reference[:, :3, :3] = axis_angle_to_matrix(torch.tensor([[0.0, 2.2, 0.0]]))
        reference[:, :3, 3] = torch.tensor([[0.2, -0.1, 0.6]])
        with torch.no_grad():
            head.rotation_projection[-1].bias.fill_(-1000.0)
            head.translation_projection[-1].bias.fill_(1000.0)
        final = head(torch.zeros(1, 8), torch.zeros(1, 3), torch.ones(1), reference_pose=reference)["base_T_C_from_O"]
        angular_delta = torch.rad2deg(rotation_geodesic_distance(final[:, :3, :3], reference[:, :3, :3]))
        translation_delta = torch.linalg.vector_norm(final[:, :3, 3] - reference[:, :3, 3], dim=-1)
        self.assertLessEqual(float(angular_delta), 5.0001)
        self.assertLessEqual(float(translation_delta), 0.0030001)


if __name__ == "__main__":
    unittest.main()
