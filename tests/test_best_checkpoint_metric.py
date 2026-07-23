from __future__ import annotations

import unittest

from symm_template_reg.engine.overfit_trainer import is_checkpoint_improvement


class BestCheckpointMetricTest(unittest.TestCase):
    def test_oracle_improvement_cannot_override_worse_top1_runtime_cost(self) -> None:
        best = {
            "eval/top1_scored_pose_cost": 0.5,
            "eval/oracle_best_pose_cost": 0.4,
            "eval/top1_pose_success_5deg_5mm": 0.5,
        }
        candidate = {
            "eval/top1_scored_pose_cost": 0.6,
            "eval/oracle_best_pose_cost": 0.1,
            "eval/top1_pose_success_5deg_5mm": 0.9,
        }
        self.assertFalse(
            is_checkpoint_improvement(
                candidate,
                best,
                metric_name="eval/top1_scored_pose_cost",
                mode="min",
                min_delta=1e-6,
                tie_breaker_name="eval/top1_pose_success_5deg_5mm",
                tie_breaker_mode="max",
            )
        )

    def test_top1_tie_uses_success_as_max_tie_breaker(self) -> None:
        best = {
            "eval/top1_scored_pose_cost": 0.5,
            "eval/top1_pose_success_5deg_5mm": 0.5,
        }
        candidate = {
            "eval/top1_scored_pose_cost": 0.5,
            "eval/top1_pose_success_5deg_5mm": 0.75,
        }
        self.assertTrue(
            is_checkpoint_improvement(
                candidate,
                best,
                metric_name="eval/top1_scored_pose_cost",
                mode="min",
                min_delta=1e-6,
                tie_breaker_name="eval/top1_pose_success_5deg_5mm",
            )
        )


if __name__ == "__main__":
    unittest.main()
