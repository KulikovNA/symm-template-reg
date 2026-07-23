import tarfile,tempfile,unittest
from pathlib import Path
from tools.package_joint_stage_report import package_joint_stage_report

class PackageTest(unittest.TestCase):
    def test_excludes_large_binary_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/"run";r.mkdir();(r/"a.json").write_text("{}");(r/"mesh.ply").write_text("ply");(r/"best.pth").write_bytes(b"x"); out=Path(td)/"report.tar.gz";package_joint_stage_report(r,out)
            with tarfile.open(out) as a: names=a.getnames()
            self.assertTrue(any(n.endswith("a.json") for n in names));self.assertFalse(any(n.endswith((".ply",".pth")) for n in names))
