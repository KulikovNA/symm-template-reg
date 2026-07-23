from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.losses import PoseSetLoss
from symm_template_reg.models.symmetry.groups import CyclicGroup
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms
from tests.test_fragment_symmetry_targets import metadata


class QueryAssignmentTest(unittest.TestCase):
    def test_queries_are_base_alternatives_and_best_symmetry_aware_query_is_assigned(self) -> None:
        sidecar = metadata()
        gt = torch.eye(4)
        gt[:3, 3] = torch.tensor([0.1, -0.03, 0.5])
        symmetry = symmetry_transforms(
            CyclicGroup(4), sidecar.axis.direction, sidecar.axis.origin
        )
        predictions = torch.eye(4).repeat(1, 8, 1, 1)
        predictions[0, :, :3, 3] = torch.tensor(
            [[0.4, 0.0, 0.5], [0.3, 0.0, 0.5], [0.2, 0.0, 0.5],
             [0.0, 0.0, 0.5], [0.1, -0.03, 0.5], [0.1, 0.2, 0.5],
             [0.1, -0.2, 0.5], [-0.1, 0.0, 0.5]]
        )
        predictions[0, 4] = gt @ symmetry[2]
        result = PoseSetLoss()(
            predictions,
            torch.zeros((1, 8)),
            gt.unsqueeze(0),
            symmetry_metadata=[sidecar],
            effective_symmetry_groups=[{"type": "C", "order": 4}],
        )
        self.assertEqual(result["assigned_query_indices"].tolist(), [4])
        # Exactly one query is assigned; the four C4 elements are not treated
        # as four independent DETR targets.
        self.assertEqual(result["assigned_query_indices"].numel(), 1)


if __name__ == "__main__":
    unittest.main()
