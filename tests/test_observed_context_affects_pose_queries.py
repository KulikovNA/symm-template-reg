from __future__ import annotations

import unittest

import torch

from symm_template_reg.engine.view_ladder import pose_context_change
from symm_template_reg.models.heads.pose_query_head import PoseQueryHead


class ObservedContextAffectsPoseQueriesTest(unittest.TestCase):
    def test_changed_memory_changes_query_poses(self) -> None:
        torch.manual_seed(4)
        head = PoseQueryHead(
            embed_dim=16,
            num_heads=4,
            num_queries=3,
            num_decoder_layers=2,
            feedforward_dim=32,
            pose_codec={"type": "PoseCodec"},
        ).eval()
        mask = torch.ones((1, 5), dtype=torch.bool)
        centroid = torch.tensor([[0.0, 0.0, 0.5]])
        scale = torch.tensor([0.1])
        first = head(torch.randn(1, 5, 16), mask, centroid, scale)
        second = head(torch.randn(1, 5, 16) + 2.0, mask, centroid, scale)
        result = pose_context_change(
            first["pose_hypotheses"],
            second["pose_hypotheses"],
            first["pose_parameters_normalized"],
            second["pose_parameters_normalized"],
        )
        self.assertEqual(result["diagnosis"], "pose_queries_respond_to_observed_context")


if __name__ == "__main__":
    unittest.main()
