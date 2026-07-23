from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.losses.region_loss import (
    active_region_binary_loss,
    masked_point_region_cross_entropy,
)


class RegionHeadSupervisionTest(unittest.TestCase):
    def test_point_ce_ignores_padding_oob_and_unused_region_slots(self) -> None:
        logits = torch.zeros((1, 4, 16), requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 0] = 8.0
            logits[0, 1, 1] = 8.0
            logits[0, 2, 3] = 8.0
        labels = torch.tensor([[0, 1, 3, -1]])
        point_valid = labels.ge(0)
        region_valid = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
        first = masked_point_region_cross_entropy(
            logits, labels, point_valid, region_valid
        )
        changed = logits.detach().clone()
        changed[..., 4:] = 1e6
        second = masked_point_region_cross_entropy(
            changed, labels, point_valid, region_valid
        )
        torch.testing.assert_close(first, second)
        first.backward()
        self.assertEqual(float(logits.grad[..., 4:].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[..., :4].abs().sum()), 0.0)

    def test_active_loss_uses_only_sidecar_slots(self) -> None:
        logits = torch.zeros((1, 16), requires_grad=True)
        target = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
        valid = torch.ones_like(target)
        loss = active_region_binary_loss(logits, target, valid)
        loss.backward()
        self.assertGreater(float(logits.grad[:, :4].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[:, 4:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
