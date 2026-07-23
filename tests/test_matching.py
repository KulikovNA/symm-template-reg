import unittest

import torch

from symm_template_reg.models.matching import HungarianPoseAssigner, LogOptimalTransport


class MatchingTest(unittest.TestCase):
    def test_rectangular_assignment(self):
        cost = torch.tensor([[0.1, 2.0], [1.0, 0.2], [0.5, 0.6]])
        prediction, target = HungarianPoseAssigner()(cost)
        self.assertEqual(prediction.tolist(), [0, 1])
        self.assertEqual(target.tolist(), [0, 1])

    def test_masked_sinkhorn_is_finite_and_differentiable(self):
        scores = torch.randn(2, 4, 5, requires_grad=True)
        row_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
        result = LogOptimalTransport(num_iterations=4)(scores, row_mask)
        self.assertEqual(tuple(result.shape), (2, 4, 5))
        self.assertTrue(torch.isfinite(result).all())
        result[row_mask].mean().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())


if __name__ == "__main__":
    unittest.main()
