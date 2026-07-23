from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.pose.weighted_procrustes import WeightedProcrustes


class WeightedProcrustesReflectionTest(unittest.TestCase):
    def test_reflection_is_corrected_to_proper_rotation(self) -> None:
        torch.manual_seed(7)
        source = torch.randn(1, 12, 3)
        target = source.clone()
        target[..., 0] *= -1
        transform = WeightedProcrustes()(
            source,
            target,
            torch.ones(1, 12),
            torch.ones(1, 12, dtype=torch.bool),
        )
        self.assertAlmostEqual(float(torch.linalg.det(transform[0, :3, :3])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
