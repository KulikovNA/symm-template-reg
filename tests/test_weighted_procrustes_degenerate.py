from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.pose.weighted_procrustes import WeightedProcrustes


class WeightedProcrustesDegenerateTest(unittest.TestCase):
    def test_collinear_subset_fails_with_rank_diagnostics(self) -> None:
        points = torch.stack((torch.linspace(0, 1, 8), torch.zeros(8), torch.zeros(8)), -1)[None]
        with self.assertRaisesRegex(ValueError, "insufficient rank.*source_rank"):
            WeightedProcrustes(fail_on_degenerate=True)(
                points, points, torch.ones(1, 8), torch.ones(1, 8, dtype=torch.bool)
            )


if __name__ == "__main__":
    unittest.main()
