import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


class PackageTrainingReportTest(unittest.TestCase):
    def test_excludes_weights_and_geometry(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            (run / "resolved_config.json").write_text("{}\n")
            (run / "best.pth").write_bytes(b"weight")
            (run / "mesh.ply").write_bytes(b"geometry")
            archive = root / "report.tar.gz"
            subprocess.run(
                [
                    sys.executable,
                    str(repo / "tools/package_training_report.py"),
                    "--input-dir", str(run),
                    "--output", str(archive),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with tarfile.open(archive) as stream:
                names = stream.getnames()
                manifest = json.load(stream.extractfile("training_report_manifest.json"))
        self.assertIn("run/resolved_config.json", names)
        self.assertNotIn("run/best.pth", names)
        self.assertNotIn("run/mesh.ply", names)
        self.assertEqual(manifest["format"], "symm-template-reg-compact-report-v1")


if __name__ == "__main__":
    unittest.main()
