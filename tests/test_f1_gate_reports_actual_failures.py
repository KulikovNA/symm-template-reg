import unittest
from symm_template_reg.evaluation.fine_stage import fine_coordinate_gate


class F1GateReportingTest(unittest.TestCase):
    def test_actual_metric_names_and_thresholds_are_reported(self):
        gate = fine_coordinate_gate({
            "aux_coordinate_p95_mm": 1.01225, "aux_coordinate_rmse_mm": .5242,
            "target_leakage_detected": False, "fine_feature_collision_fraction": 0,
            "fine_feature_variance": .3,
        })
        self.assertEqual(gate["failures"], ["aux_coordinate_p95_mm", "aux_coordinate_rmse_mm"])
        self.assertEqual(gate["thresholds"]["aux_coordinate_p95_mm"]["value"], 1.0)


if __name__ == "__main__": unittest.main()

