import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.package_four_view_coordinate_report import package_four_view_report


class FourViewPackageTest(unittest.TestCase):
    def test_binary_files_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"; root.mkdir(); (root / "a.json").write_text("{}")
            (root / "weights.pth").write_bytes(b"x"); (root / "mesh.ply").write_bytes(b"x")
            archive = Path(directory) / "report.tar.gz"
            package_four_view_report([root], archive)
            with tarfile.open(archive) as stream: names = stream.getnames()
            self.assertTrue(any(name.endswith("a.json") for name in names))
            self.assertFalse(any(name.endswith((".pth", ".ply")) for name in names))


if __name__ == "__main__": unittest.main()
