import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.package_eight_view_coordinate_report import package_eight_view_report


class EightViewReportPackageTest(unittest.TestCase):
    def test_only_compact_text_reports_are_included(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"; root.mkdir()
            for name in ("a.json", "b.csv", "c.jsonl", "d.md"):
                (root / name).write_text("{}\n")
            for name in ("x.pth", "x.ply", "x.pt", "x.npy", "x.npz"):
                (root / name).write_bytes(b"binary")
            output = Path(directory) / "report.tar.gz"
            package_eight_view_report([root], output)
            with tarfile.open(output) as archive:
                names = archive.getnames()
        self.assertTrue(any(name.endswith("a.json") for name in names))
        self.assertFalse(any(Path(name).suffix in {".pth", ".ply", ".pt", ".npy", ".npz"} for name in names))


if __name__ == "__main__":
    unittest.main()
