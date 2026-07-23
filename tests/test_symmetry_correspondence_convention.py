from __future__ import annotations

import dataclasses
import unittest

import torch

from symm_template_reg.models.losses import SymmetryAwareCorrespondenceLoss
from symm_template_reg.models.symmetry.groups import CyclicGroup, rotation_group_matrices
from symm_template_reg.models.symmetry.metadata import SymmetryAxis
from tests.test_fragment_symmetry_targets import metadata


class SymmetryCorrespondenceConventionTest(unittest.TestCase):
    def test_canonical_target_is_inverse_shared_group_action_about_axis_origin(self) -> None:
        meta = dataclasses.replace(
            metadata(),
            axis=SymmetryAxis("offset", (0.1, -0.2, 0.3), (0.0, 1.0, 0.0)),
        )
        target = torch.tensor([[[0.14, -0.18, 0.31], [0.08, -0.12, 0.35]]])
        rotation = rotation_group_matrices(CyclicGroup(4), torch.tensor(meta.axis.direction))[1]
        origin = torch.tensor(meta.axis.origin)
        prediction = torch.einsum("ij,bnj->bni", rotation.T, target - origin) + origin
        result = SymmetryAwareCorrespondenceLoss(robust_type="l2").forward_with_diagnostics(
            prediction, target, torch.ones(1, 2, dtype=torch.bool), [meta], [{"type": "C", "order": 4}]
        )
        self.assertLess(float(result["loss"]), 1e-10)
        self.assertEqual(result["selected_shared_symmetry_element"].tolist(), [1])
        self.assertTrue(torch.allclose(result["matched_target_points_O"], prediction))


if __name__ == "__main__":
    unittest.main()
