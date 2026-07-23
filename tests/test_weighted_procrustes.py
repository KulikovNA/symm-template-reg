from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.pose.rotation import axis_angle_to_matrix
from symm_template_reg.models.pose.weighted_procrustes import WeightedProcrustes


class WeightedProcrustesTest(unittest.TestCase):
    def test_recovers_synthetic_transform_and_backpropagates(self) -> None:
        torch.manual_seed(6)
        source = torch.randn(2, 20, 3, requires_grad=True)
        rotation = axis_angle_to_matrix(torch.tensor([[0.2, -0.3, 0.1]])).expand(2, 3, 3)
        translation = torch.tensor([[0.1, -0.2, 0.4], [-0.03, 0.04, 0.5]])
        target = torch.einsum("bij,bnj->bni", rotation, source.detach()) + translation[:, None]
        transform = WeightedProcrustes()(
            source,
            target,
            torch.ones(2, 20),
            torch.ones(2, 20, dtype=torch.bool),
        )
        self.assertTrue(torch.allclose(transform[:, :3, :3], rotation, atol=1e-5))
        self.assertTrue(torch.allclose(transform[:, :3, 3], translation, atol=1e-5))
        transform.square().sum().backward()
        self.assertIsNotNone(source.grad)
        self.assertTrue(torch.isfinite(source.grad).all())


if __name__ == "__main__":
    unittest.main()
