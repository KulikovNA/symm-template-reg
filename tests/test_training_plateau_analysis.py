from __future__ import annotations

import unittest
from pathlib import Path

from tools.analyze_training_plateau import analyze_run


RUN = Path("/home/nikita/disser/fragment-template-registration-lab/work_dirs/conditioned_v2_01_k1_direct_equal_exposure_frames04_05_02_08_seed0_20260719_101204")


@unittest.skipUnless((RUN / "history/history.jsonl").is_file(), "stopped direct run unavailable")
class TrainingPlateauAnalysisTest(unittest.TestCase):
    def test_stopped_run_has_fixed_diagnosis(self) -> None:
        summary, rows = analyze_run(RUN)
        self.assertEqual(summary["status"], "plateau_with_rotation_context_collapse")
        self.assertEqual(summary["diagnosis"], "rotation_context_collapse")
        self.assertFalse(summary["continuing_same_training_recommended"])
        self.assertGreater(summary["updates_after_best_checkpoint"], 2200)
        self.assertGreaterEqual(len(rows), 10)


if __name__ == "__main__":
    unittest.main()
