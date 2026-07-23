from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/view_ladder/frames04_05_02_08.json")


@unittest.skipUnless(MANIFEST.is_file(), "real four-view manifest unavailable")
class OracleWeightedProcrustesRealSamplesTest(unittest.TestCase):
    def test_real_samples_meet_oracle_precision_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "oracle"
            subprocess.run(
                [sys.executable, str(ROOT / "tools/test_oracle_correspondence_pose.py"), "--manifest", str(MANIFEST), "--output-dir", str(output), "--device", "cpu"],
                cwd=ROOT, check=True, capture_output=True, text=True,
            )
            result = json.loads((output / "oracle_procrustes_results.json").read_text())
        self.assertTrue(result["criterion_passed"])
        self.assertLess(result["max_rotation_error_deg"], 1e-4)
        self.assertLess(result["max_translation_error_mm"], 1e-4)


if __name__ == "__main__":
    unittest.main()
