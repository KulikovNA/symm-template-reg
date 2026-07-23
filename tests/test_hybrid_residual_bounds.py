from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.heads.conditioned_base_pose_head import ConditionedBasePoseHead
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance
from symm_template_reg.evaluation.diagnostic_gates import evaluate_correspondence_diagnostic_gates


class HybridResidualBoundsTest(unittest.TestCase):
    def test_large_logits_stay_within_five_degrees_and_three_mm(self) -> None:
        head = ConditionedBasePoseHead(embed_dim=8, hidden_dim=16, split_rotation_translation=True, output_mode="bounded_correction", max_rotation_correction_deg=5.0, max_translation_correction_m=0.003)
        with torch.no_grad():
            head.rotation_projection[-1].bias.fill_(1000.0)
            head.translation_projection[-1].bias.fill_(1000.0)
        result = head(torch.zeros(2, 8), torch.zeros(2, 3), torch.ones(2), reference_pose=torch.eye(4).repeat(2, 1, 1))
        correction = result["base_correction_transform"]
        angle = torch.rad2deg(rotation_geodesic_distance(correction[:, :3, :3], torch.eye(3)))
        translation = torch.linalg.vector_norm(correction[:, :3, 3], dim=-1)
        self.assertLessEqual(float(angle.max()), 5.0001)
        self.assertLessEqual(float(translation.max()), 0.0030001)

    def test_static_identity_is_not_mislabeled_as_static_codebook(self) -> None:
        config = {
            "enabled": True,
            "residual_static_codebook": {
                "enabled": True,
                "max_static_fraction": 0.25,
                "minimum_nonidentity_rotation_deg": 0.1,
                "minimum_nonidentity_translation_mm": 0.1,
            },
        }
        result = evaluate_correspondence_diagnostic_gates(
            {
                "eval/hybrid_residual_static_fraction": 1.0,
                "eval/hybrid_residual_rotation_deg": 0.0,
                "eval/hybrid_residual_translation_mm": 0.0,
            },
            config,
            min_sample_exposures=100,
        )
        self.assertFalse(result["failed"])


if __name__ == "__main__":
    unittest.main()
