import unittest

from symm_template_reg.evaluation.local_stage import check_local_substage


def passing_metrics():
    return {
        "valid_triangle_set_top1": 0.95,
        "valid_triangle_set_top4": 0.995,
        "valid_triangle_candidate_recall": 1.0,
        "local_triangle_set_ce": 0.5,
        "local_triangle_random_ce": 3.0,
        "triangle_target_index_mismatch_fraction": 0.0,
        "duplicate_local_candidate_fraction": 0.0,
        "teacher_forcing_selected_symmetry_element": 1,
        "min_local_candidate_count": 32,
        "max_local_candidate_count": 32,
        "invalid_candidate_count_fraction": 0.0,
    }


class B1StageGateNumericSemanticsTest(unittest.TestCase):
    def gate(self, metrics):
        return check_local_substage(
            "B1",
            metrics,
            nonfinite_detected=False,
            target_leakage_detected=False,
        )

    def test_fraction_just_below_one_passes_with_tolerance(self):
        metrics = passing_metrics()
        metrics["valid_triangle_candidate_recall"] = 0.99999994
        gate = self.gate(metrics)
        self.assertTrue(gate["checks"]["valid_triangle_candidate_recall"])
        self.assertTrue(gate["stage_passed"])

    def test_float_mean_is_not_used_as_an_integer_count(self):
        metrics = passing_metrics()
        metrics["mean_local_candidate_count"] = 31.999998
        gate = self.gate(metrics)
        self.assertTrue(gate["checks"]["min_local_candidate_count_is_32"])
        self.assertTrue(gate["checks"]["max_local_candidate_count_is_32"])
        self.assertNotIn("candidate_count_is_32", gate["checks"])
        self.assertTrue(gate["stage_passed"])

    def test_near_integer_float_is_rejected_for_integer_count(self):
        metrics = passing_metrics()
        metrics["min_local_candidate_count"] = 31.999998
        gate = self.gate(metrics)
        self.assertFalse(gate["checks"]["min_local_candidate_count_is_32"])


if __name__ == "__main__":
    unittest.main()
