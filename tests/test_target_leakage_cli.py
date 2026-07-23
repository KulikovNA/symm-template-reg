from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TargetLeakageCliTest(unittest.TestCase):
    def test_missing_checkpoint_fails_before_creating_output_directory(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must_not_exist"
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "tools/audit_target_leakage.py"),
                    "--config",
                    "unused.py",
                    "--manifest",
                    "unused.json",
                    "--checkpoint",
                    "/absolute/path/to/best_correspondence_only.pth",
                    "--output-dir",
                    str(output),
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Train correspondence-only first", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
