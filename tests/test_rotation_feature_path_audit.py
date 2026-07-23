from __future__ import annotations

import unittest

import torch

from symm_template_reg.evaluation.rotation_feature_path import (
    centered_cloud_chamfer_matrix,
    vector_pairwise_matrix,
)


class RotationFeaturePathAuditTest(unittest.TestCase):
    def test_pairwise_audit_preserves_view_dimension_and_detects_variation(self) -> None:
        clouds = torch.randn(4, 12, 3)
        clouds[1:, :, 0] += torch.arange(1, 4)[:, None]
        mask = torch.ones(4, 12, dtype=torch.bool)
        raw = centered_cloud_chamfer_matrix(clouds, mask)
        tokens = vector_pairwise_matrix(clouds.flatten(1))
        self.assertEqual(tuple(raw.shape), (4, 4))
        self.assertEqual(tuple(tokens.shape), (4, 4))
        self.assertGreater(float(tokens[0, 3]), 0.0)


if __name__ == "__main__":
    unittest.main()
