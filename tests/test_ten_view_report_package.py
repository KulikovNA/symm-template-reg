import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.package_ten_view_scratch_report import package_ten_view_scratch_report


class TenViewReportPackageTest(unittest.TestCase):
    def test_only_compact_text_reports_are_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"; root.mkdir()
            for name in ("a.json", "b.csv", "c.jsonl", "d.md"):
                (root / name).write_text("{}\n", encoding="utf-8")
            for name in ("checkpoint.pth", "mesh.ply", "tensor.pt", "x.npy", "x.npz"):
                (root / name).write_bytes(b"binary")
            (root / "ranking_diagnostics.json").write_text("{}\n", encoding="utf-8")
            output = Path(directory) / "report.tar.gz"
            report = package_ten_view_scratch_report([root], output)
            with tarfile.open(output) as archive:
                names = archive.getnames()
            self.assertEqual(report["file_count"], 4)
            self.assertFalse(any(Path(name).suffix in {".pth", ".ply", ".pt", ".npy", ".npz"} for name in names))
            self.assertIn("packaging_manifest.json", names)


if __name__ == "__main__": unittest.main()
