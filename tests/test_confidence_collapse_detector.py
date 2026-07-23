from __future__ import annotations

import unittest

from symm_template_reg.evaluation.diagnostic_gates import evaluate_correspondence_diagnostic_gates


class ConfidenceCollapseDetectorTest(unittest.TestCase):
    def test_low_effective_count_has_distinct_diagnosis(self) -> None:
        result = evaluate_correspondence_diagnostic_gates(
            {"eval/effective_correspondence_count": 1.2, "eval/correspondence_context_pairwise_distance": 1.0, "eval/correspondence_pose_rank_valid": 1.0},
            {"enabled": True, "min_sample_exposures": 10, "confidence_collapse": {"enabled": True, "minimum_effective_point_count": 16}},
            min_sample_exposures=100,
        )
        self.assertTrue(result["failed"])
        self.assertEqual(result["diagnosis"], "confidence_collapse")


if __name__ == "__main__":
    unittest.main()
