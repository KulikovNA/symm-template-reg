from __future__ import annotations

import unittest

from symm_template_reg.engine.view_ladder import view_scaling_summary


class ViewScalingMetricsTest(unittest.TestCase):
    def test_requested_curve_metrics(self) -> None:
        rows = [
            {
                "oracle_topk_rotation_error_deg": 1.0,
                "oracle_translation_total_mm": 1.5,
                "oracle_topk_success_2deg_2mm": "True",
            },
            {
                "oracle_topk_rotation_error_deg": 3.0,
                "oracle_translation_total_mm": 2.5,
                "oracle_topk_success_2deg_2mm": "False",
            },
        ]
        summary = view_scaling_summary(rows, num_queries=8)
        self.assertEqual(summary["num_views"], 2)
        self.assertEqual(summary["K"], 8)
        self.assertEqual(summary["mean_rotation_deg"], 2.0)
        self.assertEqual(summary["pose_success_2deg_2mm"], 0.5)


if __name__ == "__main__":
    unittest.main()
