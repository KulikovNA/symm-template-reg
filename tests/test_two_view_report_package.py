import tarfile,tempfile,unittest
from pathlib import Path
from tools.package_two_view_coordinate_report import package
class PackageTest(unittest.TestCase):
    def test_binary_files_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();(root/'a.json').write_text('{}');(root/'x.pth').write_bytes(b'x');out=Path(d)/'r.tar.gz';package([root],out)
            with tarfile.open(out) as a:names=a.getnames()
            self.assertTrue(any(x.endswith('.json') for x in names));self.assertFalse(any(x.endswith('.pth') for x in names))
