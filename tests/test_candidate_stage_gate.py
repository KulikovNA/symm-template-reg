import unittest

from symm_template_reg.evaluation.patch_stage import candidate_stage_gate


class CandidateStageGateTest(unittest.TestCase):
    def test_candidate_gate_passes_without_top1_quality(self):
        gate = candidate_stage_gate(
            {
                "valid_patch_set_top1_accuracy": 0.7,
                "valid_patch_set_top4_recall": 1.0,
                "valid_patch_set_in_candidate_set_fraction": 1.0,
                "unique_predicted_patches": 19,
                "most_popular_patch_fraction": 0.26,
            },
            nonfinite_detected=False,
            target_leakage_detected=False,
            capacity_audit_passed=True,
        )
        self.assertTrue(gate["candidate_stage_passed"])

    def test_patch_collapse_fails(self):
        gate = candidate_stage_gate(
            {
                "valid_patch_set_top4_recall": 1.0,
                "valid_patch_set_in_candidate_set_fraction": 1.0,
                "unique_predicted_patches": 1,
                "most_popular_patch_fraction": 1.0,
            },
            nonfinite_detected=False,
            target_leakage_detected=False,
            capacity_audit_passed=True,
        )
        self.assertFalse(gate["candidate_stage_passed"])


if __name__ == "__main__":
    unittest.main()
