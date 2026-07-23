from __future__ import annotations

import unittest

import torch

from symm_template_reg.evaluation.context_conditioning import (
    context_conditioning_metrics,
)
from symm_template_reg.models.pose.pose_representation import make_transform
from symm_template_reg.models.pose.rotation import axis_angle_to_matrix


class StaticCodebookDetectionTest(unittest.TestCase):
    def test_constant_prediction_is_detected_against_changing_gt(self) -> None:
        identity = torch.eye(4)
        rotated = make_transform(
            axis_angle_to_matrix(
                torch.tensor([[0.0, 0.0, torch.pi / 2]])
            )[0],
            torch.zeros(3),
        )
        rows = [
            {
                "base_T_C_from_O": identity.tolist(),
                "gt_T_C_from_O": identity.tolist(),
                "query_T_C_from_O": [identity.tolist(), identity.tolist()],
                "sample_context": [0.0, 0.0],
            },
            {
                "base_T_C_from_O": identity.tolist(),
                "gt_T_C_from_O": rotated.tolist(),
                "query_T_C_from_O": [identity.tolist(), identity.tolist()],
                "sample_context": [1.0, 0.0],
            },
        ]
        metrics = context_conditioning_metrics(rows)
        self.assertEqual(metrics["base_pose_static_fraction"], 1.0)
        self.assertEqual(metrics["query_static_codebook_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
