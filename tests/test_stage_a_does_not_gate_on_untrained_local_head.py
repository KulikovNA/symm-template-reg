import unittest

from symm_template_reg.config import load_config
from symm_template_reg.evaluation.patch_stage import candidate_stage_gate


class StageAGateIsolationTest(unittest.TestCase):
    def test_bad_local_and_pose_diagnostics_are_not_inputs(self):
        gate = candidate_stage_gate(
            {
                "valid_patch_set_top4_recall": 1.0,
                "valid_patch_set_in_candidate_set_fraction": 1.0,
                "unique_predicted_patches": 4,
                "most_popular_patch_fraction": 0.4,
                "triangle_top1_accuracy": 0.0,
                "barycentric_reconstruction_p95_mm": 100.0,
                "rotation_error_deg": 180.0,
            },
            nonfinite_detected=False,
            target_leakage_detected=False,
            capacity_audit_passed=True,
        )
        self.assertTrue(gate["candidate_stage_passed"])

    def test_stage_a_disables_rank_collapse_early_stop(self):
        for frame in ("04", "08"):
            config = load_config(
                f"configs/debug/correspondence_head_v4/"
                f"00_patch_classifier_frame{frame}.py"
            )
            self.assertEqual(config["train"]["rank_collapse_patience_evals"], 0)


if __name__ == "__main__":
    unittest.main()
