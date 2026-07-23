from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.losses import PoseSetLoss
from tests.test_fragment_symmetry_targets import metadata


class TrainingTargetOracleTest(unittest.TestCase):
    def test_perfect_pose_and_classification_have_near_zero_training_loss(self) -> None:
        gt = torch.eye(4).unsqueeze(0)
        predictions = torch.eye(4).repeat(1, 8, 1, 1)
        predictions[0, 1:, :3, 3] = 0.25
        logits = torch.full((1, 8), -20.0)
        logits[0, 0] = 20.0
        result = PoseSetLoss()(
            predictions,
            logits,
            gt,
            symmetry_metadata=[metadata()],
            effective_symmetry_groups=[{"type": "C", "order": 4}],
        )
        self.assertEqual(result["assigned_query_indices"].tolist(), [0])
        self.assertLess(float(result["loss_pose_set"]), 1e-3)


if __name__ == "__main__":
    unittest.main()
