from __future__ import annotations

import unittest

import torch

from symm_template_reg.engine.single_fragment import world_pose_consistency
from symm_template_reg.models.symmetry.metadata import (
    SymmetryAxis,
    SymmetryMetadata,
)


class CrossViewWorldConsistencyTest(unittest.TestCase):
    def test_world_composition_and_zero_spread(self):
        T_W_from_C = torch.eye(4)
        T_W_from_C[0, 3] = 2.0
        T_C_from_O = torch.eye(4)
        T_C_from_O[1, 3] = 3.0
        expected = T_W_from_C @ T_C_from_O
        self.assertEqual(expected[:3, 3].tolist(), [2.0, 3.0, 0.0])
        metadata = SymmetryMetadata(
            version=1,
            object_model_id="object",
            coordinate_frame="O",
            axis=SymmetryAxis("axis", (0, 0, 0), (0, 0, 1)),
            regions=(),
        )
        metrics = world_pose_consistency(
            expected.unsqueeze(0).repeat(10, 1, 1), metadata, {"type": "C", "order": 1}
        )
        self.assertTrue(all(abs(value) < 1e-8 for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
