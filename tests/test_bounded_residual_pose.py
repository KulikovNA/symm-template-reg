import unittest

import torch

from symm_template_reg.models.heads.residual_pose_hypothesis_head import ResidualPoseHypothesisHead
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance


class BoundedResidualPoseTest(unittest.TestCase):
    def test_cannot_encode_180_degree_absolute_pose(self):
        head = ResidualPoseHypothesisHead(
            embed_dim=8, num_heads=2, num_hypotheses=2, num_decoder_layers=1,
            feedforward_dim=16,
            residual_bounds={"max_rotation_deg": 15.0, "max_translation_m": 0.01},
        )
        with torch.no_grad():
            head.residual_projection[-1].bias.fill_(1000.0)
        output = head(torch.zeros(1, 8), torch.zeros(1, 3, 8), torch.ones(1, 3, dtype=torch.bool), torch.eye(4)[None], torch.ones(1))
        angle = torch.rad2deg(rotation_geodesic_distance(output["residual_transforms"][0, :, :3, :3], torch.eye(3)))
        self.assertLessEqual(float(angle.max()), 15.0001)


if __name__ == "__main__": unittest.main()
