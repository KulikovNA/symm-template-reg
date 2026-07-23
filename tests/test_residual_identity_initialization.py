from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.heads.residual_pose_hypothesis_head import (
    ResidualPoseHypothesisHead,
)


class ResidualIdentityInitializationTest(unittest.TestCase):
    def test_initial_residuals_are_near_identity(self) -> None:
        torch.manual_seed(3)
        head = ResidualPoseHypothesisHead(
            embed_dim=16,
            num_heads=4,
            num_hypotheses=4,
            num_decoder_layers=1,
            feedforward_dim=32,
        ).eval()
        result = head(
            torch.randn(2, 16),
            torch.randn(2, 5, 16),
            torch.ones(2, 5, dtype=torch.bool),
            torch.eye(4).expand(2, 4, 4),
            torch.ones(2),
        )
        identity = torch.eye(4).view(1, 1, 4, 4)
        self.assertLess(
            float((result["residual_transforms"] - identity).abs().max().detach()), 0.02
        )


if __name__ == "__main__":
    unittest.main()
