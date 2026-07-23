import json
import tempfile
import unittest
from pathlib import Path

from symm_template_reg.evaluation.patch_stage import write_patch_stage_gates


class PatchStageReportingTest(unittest.TestCase):
    def test_legacy_gate_links_both_nonempty_threshold_reports(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            _, _, legacy = write_patch_stage_gates(
                output,
                {
                    "valid_patch_set_top1_accuracy": 0.9,
                    "valid_patch_set_top4_recall": 1.0,
                    "valid_patch_set_in_candidate_set_fraction": 1.0,
                    "unique_predicted_patches": 5,
                    "most_popular_patch_fraction": 0.3,
                },
                nonfinite_detected=False,
                target_leakage_detected=False,
                capacity_audit_passed=True,
            )
            self.assertTrue(legacy["stage_passed"])
            self.assertTrue(legacy["thresholds"])
            self.assertTrue(Path(legacy["candidate_stage_gate_path"]).is_file())
            self.assertTrue(Path(legacy["top1_quality_gate_path"]).is_file())
            self.assertFalse(json.loads((output / "top1_quality_gate.json").read_text())["top1_quality_passed"])


if __name__ == "__main__":
    unittest.main()
