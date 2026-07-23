from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.losses import SymmetryAwareCorrespondenceLoss
from symm_template_reg.models.symmetry.groups import CyclicGroup, rotation_group_matrices
from tests.test_fragment_symmetry_targets import metadata


class SharedGroupElementCorrespondenceTest(unittest.TestCase):
    def test_independent_per_point_sectors_are_rejected_by_loss(self) -> None:
        target = torch.tensor([[[0.02, 0.0, 0.01], [-0.03, 0.01, 0.02]]])
        rotation = rotation_group_matrices(CyclicGroup(2), torch.tensor(metadata().axis.direction))[1]
        prediction = target.clone()
        prediction[:, 1] = torch.einsum("ij,bj->bi", rotation, target[:, 1])
        loss = SymmetryAwareCorrespondenceLoss(robust_type="l1")(
            prediction, target, torch.ones(1, 2, dtype=torch.bool), [metadata()], [{"type": "C", "order": 2}]
        )
        self.assertGreater(float(loss), 0.01)


if __name__ == "__main__":
    unittest.main()
