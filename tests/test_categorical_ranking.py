from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.losses import PoseQueryRankingLoss


class CategoricalRankingTest(unittest.TestCase):
    def test_oracle_query_is_argmin_pose_cost(self):
        logits = torch.tensor([[0.0, 0.0, 3.0]], requires_grad=True)
        costs = torch.tensor([[2.0, 1.0, 0.1]])
        result = PoseQueryRankingLoss(type="matched_categorical")(logits, costs)
        self.assertEqual(int(result["oracle_query_indices"][0]), 2)
        result["loss_pose_query_ranking"].backward()
        self.assertLess(float(logits.grad[0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
