import unittest
from symm_template_reg.evaluation.fine_stage import fine_coordinate_gate


class FineCoordinateGateTest(unittest.TestCase):
    def test_gate(self):
        self.assertTrue(fine_coordinate_gate({"aux_coordinate_p95_mm": 1, "aux_coordinate_rmse_mm": .5, "fine_feature_variance": .01, "target_leakage_detected": False})["passed"])


if __name__ == "__main__": unittest.main()

