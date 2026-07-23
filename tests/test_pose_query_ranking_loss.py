from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.losses import PoseQueryRankingLoss


class PoseQueryRankingLossTest(unittest.TestCase):
    def test_soft_quality_target_prefers_lower_pose_cost(self) -> None:
        costs = torch.tensor([[0.0, 0.4, 1.0]])
        criterion = PoseQueryRankingLoss(type="soft_quality", temperature=0.25)
        target = criterion.target_distribution(costs)
        self.assertGreater(float(target[0, 0]), float(target[0, 1]))
        self.assertGreater(float(target[0, 1]), float(target[0, 2]))
        good = criterion(torch.tensor([[5.0, 0.0, -2.0]]), costs)[
            "loss_pose_query_ranking"
        ]
        bad = criterion(torch.tensor([[-2.0, 0.0, 5.0]]), costs)[
            "loss_pose_query_ranking"
        ]
        self.assertLess(float(good), float(bad))

    def test_cost_target_is_detached_by_default(self) -> None:
        costs = torch.tensor([[0.2, 0.1]], requires_grad=True)
        logits = torch.zeros((1, 2), requires_grad=True)
        loss = PoseQueryRankingLoss()(logits, costs)["loss_pose_query_ranking"]
        loss.backward()
        self.assertIsNone(costs.grad)
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
