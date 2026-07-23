from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.losses import SymmetryAwareCorrespondenceLoss
from symm_template_reg.models.symmetry.groups import (
    CyclicGroup,
    rotation_group_matrices,
)
from tests.test_fragment_symmetry_targets import metadata


class SymmetryAwareCorrespondenceLossTest(unittest.TestCase):
    def test_c2_c4_c10_accept_one_shared_equivalent_target(self) -> None:
        target = torch.tensor(
            [[[0.01, -0.02, 0.00], [0.02, 0.01, 0.03], [-0.01, 0.03, 0.02]]]
        )
        axis = torch.tensor(metadata().axis.direction)
        for order in (2, 4, 10):
            rotation = rotation_group_matrices(CyclicGroup(order), axis)[1]
            prediction = torch.einsum("ij,bnj->bni", rotation, target)
            loss = SymmetryAwareCorrespondenceLoss()(
                prediction,
                target,
                torch.ones(1, 3, dtype=torch.bool),
                [metadata()],
                [{"type": "C", "order": order}],
            )
            self.assertLess(float(loss), 1e-8)


if __name__ == "__main__":
    unittest.main()
