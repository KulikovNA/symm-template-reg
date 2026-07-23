import tarfile, tempfile, unittest
from pathlib import Path
from tools.package_four_fragments_four_views_report import package
class TestPackage(unittest.TestCase):
    def test_excludes_weights_and_geometry(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/"source"; source.mkdir(); (source/"a.json").write_text("{}\n"); (source/"x.pth").write_bytes(b"x"); (source/"x.ply").write_text("ply\n")
            out=Path(root)/"r.tar.gz"; package([source],out)
            with tarfile.open(out) as archive: names=archive.getnames()
            self.assertTrue(any(x.endswith("a.json") for x in names)); self.assertFalse(any(x.endswith((".pth",".ply")) for x in names))

