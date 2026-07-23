from __future__ import annotations

import unittest

from symm_template_reg.evaluation.plateau import detect_rotation_context_plateau


def records(static: bool) -> list[dict[str, float]]:
    return [
        {
            "eval/oracle_best_pose_cost": 1.0,
            "eval/rotation_response_ratio": 1e-6 if static else 0.5,
            "eval/base_pose_pairwise_rotation_deg": 1e-4 if static else 30.0,
            "eval/gt_pose_pairwise_rotation_deg": 78.5,
            "eval/base_pose_static_fraction": 1.0 if static else 0.0,
        }
        for _ in range(11)
    ]


class RotationContextCollapseDetectorTest(unittest.TestCase):
    def test_plateau_alone_does_not_stop(self) -> None:
        self.assertFalse(detect_rotation_context_plateau(records(False), min_sample_exposures=400)["detected"])

    def test_plateau_plus_static_rotation_stops_with_diagnosis(self) -> None:
        result = detect_rotation_context_plateau(records(True), min_sample_exposures=400)
        self.assertTrue(result["detected"])
        self.assertEqual(result["diagnosis"], "rotation_context_collapse")


if __name__ == "__main__":
    unittest.main()
