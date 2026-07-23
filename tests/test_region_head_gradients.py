from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.heads.symmetry_region_head import SymmetryRegionHead
from symm_template_reg.models.losses.region_loss import (
    active_region_binary_loss,
    masked_point_region_cross_entropy,
)


class RegionHeadGradientTest(unittest.TestCase):
    def test_both_region_branches_receive_nonzero_gradients(self) -> None:
        torch.manual_seed(4)
        head = SymmetryRegionHead(embed_dim=8, max_regions=16)
        point_logits, active_logits = head(
            torch.randn(2, 6, 8),
            torch.randn(2, 8),
            torch.randn(2, 8),
            torch.ones((2, 6), dtype=torch.bool),
        )
        labels = torch.tensor([[0, 1, 2, 3, 0, 1], [3, 2, 1, 0, 3, 2]])
        point_valid = torch.ones_like(labels, dtype=torch.bool)
        region_valid = torch.ones((2, 4), dtype=torch.bool)
        active_target = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=torch.bool)
        loss = masked_point_region_cross_entropy(
            point_logits, labels, point_valid, region_valid
        ) + active_region_binary_loss(
            active_logits, active_target, region_valid
        )
        loss.backward()
        self.assertGreater(float(head.point_classifier.weight.grad.norm()), 0.0)
        self.assertGreater(float(head.active_classifier[-1].weight.grad.norm()), 0.0)
        self.assertEqual(float(head.point_classifier.weight.grad[4:].abs().sum()), 0.0)
        self.assertEqual(float(head.active_classifier[-1].weight.grad[4:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
