from __future__ import annotations

import unittest

import torch

from symm_template_reg.models import register_all_modules
from symm_template_reg.registry import GEOMETRY_MODULES, build_from_cfg


class DualStreamGeometryEncoderTest(unittest.TestCase):
    def test_matching_and_pose_streams_are_separate_and_masked(self) -> None:
        register_all_modules()
        module = build_from_cfg(
            dict(
                type="DualStreamGeometryEncoder",
                embed_dim=8,
                matching_geometric_embedding=dict(
                    type="GeometricStructureEmbedding", embed_dim=8, num_neighbors=2
                ),
            ),
            GEOMETRY_MODULES,
        )
        features = torch.randn(1, 5, 8)
        points = torch.randn(1, 5, 3)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
        result = module(features, points, mask)
        self.assertEqual(result["matching_features"].shape, features.shape)
        self.assertEqual(result["pose_features"].shape, features.shape)
        self.assertFalse(
            torch.allclose(
                result["matching_features"][:, :3], result["pose_features"][:, :3]
            )
        )
        self.assertEqual(
            float(result["matching_features"][:, 3:].abs().sum().detach()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
