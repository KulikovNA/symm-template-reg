from __future__ import annotations

import unittest

import torch

from symm_template_reg.engine.metrics import score_pose_cost_spearman
from symm_template_reg.engine.single_fragment import ranking_diagnostics
from tools.audit_ranking_targets import _parse_metric_value


class RankingTargetDiagnosticsTest(unittest.TestCase):
    def test_csv_boolean_metric_is_parsed_as_zero_or_one(self):
        self.assertEqual(_parse_metric_value("True"), 1.0)
        self.assertEqual(_parse_metric_value("False"), 0.0)
        self.assertEqual(_parse_metric_value("0.25"), 0.25)

    def test_positive_spearman_means_high_score_low_cost(self):
        score = torch.tensor([1.0, 2.0, 3.0])
        cost = torch.tensor([3.0, 2.0, 1.0])
        self.assertAlmostEqual(score_pose_cost_spearman(score, cost), 1.0, places=6)

    def test_entropy_and_cost_statistics(self):
        values = ranking_diagnostics(
            torch.tensor([[0.0, 2.0]]),
            torch.tensor([[3.0, 1.0]]),
            torch.tensor([[0.0, 1.0]]),
        )
        self.assertEqual(float(values["pose_cost_min"][0]), 1.0)
        self.assertEqual(float(values["ranking_target_entropy"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
