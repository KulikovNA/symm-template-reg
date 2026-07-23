from __future__ import annotations

import types
import unittest

import torch

from symm_template_reg.engine.metrics import batch_pose_metric_rows
from tests.test_fragment_symmetry_targets import metadata


class Top1VersusOracleMetricTest(unittest.TestCase):
    def _batch(self):
        return {
            "sample_id": ["sample"],
            "scene_id": ["scene"],
            "fragment_id": torch.tensor([0]),
            "template_symmetry_metadata": [metadata()],
            "gt": {
                "T_C_from_O": torch.eye(4)[None],
                "effective_symmetry_group": [{"type": "C", "order": 1}],
                "active_symmetry_regions": None,
                "active_symmetry_regions_valid_mask": None,
            },
            "meta": [
                {
                    "fragment_mesh": {
                        "num_faces": 900,
                        "surface_area_m2": 1.0,
                        "bbox_diagonal_m": 1.0,
                    }
                }
            ],
        }

    def test_wrong_score_order_separates_top1_from_oracle(self) -> None:
        poses = torch.eye(4).repeat(1, 2, 1, 1)
        poses[0, 1, 0, 3] = 0.1
        prediction = types.SimpleNamespace(
            pose_hypotheses=poses,
            pose_logits=torch.tensor([[0.0, 5.0]]),
            active_region_logits=None,
            observed_region_logits=None,
        )
        row = batch_pose_metric_rows(prediction, self._batch())[0]
        self.assertGreater(row["top1_scored_pose_cost"], row["oracle_best_pose_cost"])
        self.assertGreater(row["ranking_regret"], 0.0)
        self.assertFalse(row["top1_query_is_oracle"])

    def test_correct_score_order_has_zero_regret(self) -> None:
        poses = torch.eye(4).repeat(1, 2, 1, 1)
        poses[0, 1, 0, 3] = 0.1
        prediction = types.SimpleNamespace(
            pose_hypotheses=poses,
            pose_logits=torch.tensor([[5.0, 0.0]]),
            active_region_logits=None,
            observed_region_logits=None,
        )
        row = batch_pose_metric_rows(prediction, self._batch())[0]
        self.assertAlmostEqual(row["ranking_regret"], 0.0)
        self.assertTrue(row["top1_query_is_oracle"])


if __name__ == "__main__":
    unittest.main()
