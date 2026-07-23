from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.losses.correspondence_confidence_loss import (
    CorrespondenceConfidenceRegularizationLoss,
    correspondence_confidence_diagnostics,
)


class ConfidenceEffectiveSampleCountTest(unittest.TestCase):
    def test_uniform_has_full_ess_and_one_point_collapse_is_penalized(self) -> None:
        mask = torch.ones(1, 8, dtype=torch.bool)
        uniform = correspondence_confidence_diagnostics(torch.ones(1, 8), mask)
        collapsed_weights = torch.zeros(1, 8); collapsed_weights[:, 0] = 1.0
        collapsed = correspondence_confidence_diagnostics(collapsed_weights, mask)
        loss = CorrespondenceConfidenceRegularizationLoss(minimum_effective_point_count=4)(collapsed_weights, mask)
        self.assertAlmostEqual(float(uniform["effective_count"]), 8.0, places=5)
        self.assertAlmostEqual(float(collapsed["effective_count"]), 1.0, places=5)
        self.assertGreater(float(loss["loss_confidence_regularization"]), 0.0)


if __name__ == "__main__":
    unittest.main()
