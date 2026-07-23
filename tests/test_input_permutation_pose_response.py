from __future__ import annotations

import unittest

import torch

from symm_template_reg.evaluation.context_conditioning import (
    input_permutation_equivariance_error,
)
from symm_template_reg.models import build_model
from tests.conditioned_test_utils import conditioned_batch, tiny_conditioned_config


class InputPermutationPoseResponseTest(unittest.TestCase):
    def test_observed_batch_permutation_permutes_base_pose(self) -> None:
        torch.manual_seed(4)
        model = build_model(tiny_conditioned_config()).eval()
        batch = conditioned_batch()
        with torch.no_grad():
            original = model(batch)
            changed = conditioned_batch()
            changed["observed"]["points_C"] = changed["observed"]["points_C"].flip(0)
            changed["observed"]["valid_mask"] = changed["observed"]["valid_mask"].flip(0)
            permuted = model(changed)
        metrics = input_permutation_equivariance_error(
            original.base_pose, permuted.base_pose, [1, 0]
        )
        self.assertLess(
            metrics["input_permutation_equivariance_rotation_error_deg"], 1e-3
        )
        self.assertLess(
            metrics["input_permutation_equivariance_translation_error_mm"], 1e-3
        )
        self.assertFalse(torch.allclose(original.base_pose[0], original.base_pose[1]))


if __name__ == "__main__":
    unittest.main()
