from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.heads.pose_query_head import PoseQueryHead
from symm_template_reg.models.losses.pose_set_loss import PoseSetLoss
from tests.test_fragment_symmetry_targets import metadata


class AuxiliaryDecoderPoseLossTest(unittest.TestCase):
    def test_auxiliary_pose_receives_symmetry_aware_gradient(self) -> None:
        torch.manual_seed(3)
        head = PoseQueryHead(
            embed_dim=16,
            num_heads=4,
            num_queries=2,
            num_decoder_layers=3,
            feedforward_dim=32,
        )
        output = head(torch.randn(1, 6, 16), torch.ones(1, 6, dtype=torch.bool))
        auxiliary_pose = output["auxiliary_outputs"][0]["pose_hypotheses"]
        auxiliary_pose.retain_grad()
        criterion = PoseSetLoss(
            classification_weight=0.0, auxiliary_weight=0.5
        )
        result = criterion(
            output["pose_hypotheses"],
            output["pose_logits"],
            torch.eye(4).unsqueeze(0),
            output["auxiliary_outputs"],
            symmetry_metadata=[metadata()],
            effective_symmetry_groups=[{"type": "C", "order": 2}],
        )
        result["loss_pose_set"].backward()
        self.assertGreater(float(result["loss_pose_auxiliary"].detach()), 0.0)
        self.assertIsNotNone(auxiliary_pose.grad)
        self.assertGreater(float(auxiliary_pose.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
