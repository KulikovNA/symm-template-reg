from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.pose.rotation import axis_angle_to_matrix
from symm_template_reg.models.pose.weighted_procrustes import WeightedProcrustes


class WeightedProcrustesMaskedTest(unittest.TestCase):
    def test_padding_and_zero_weight_points_cannot_change_pose(self) -> None:
        torch.manual_seed(4)
        source = torch.randn(1, 20, 3, dtype=torch.float64)
        rotation = axis_angle_to_matrix(torch.tensor([[0.2, -0.3, 0.1]], dtype=torch.float64))[0]
        translation = torch.tensor([0.01, -0.02, 0.4], dtype=torch.float64)
        target = torch.einsum("ij,bnj->bni", rotation, source) + translation
        mask = torch.zeros(1, 20, dtype=torch.bool); mask[:, :12] = True
        weights = torch.rand(1, 20, dtype=torch.float64); weights[:, 12:] = 0.0
        target[:, 12:] = 1e6
        result = WeightedProcrustes(fail_on_degenerate=True).solve(source, target, weights, mask)
        self.assertTrue(torch.allclose(result["transform"][0, :3, :3], rotation, atol=1e-10))
        self.assertTrue(torch.allclose(result["transform"][0, :3, 3], translation, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
