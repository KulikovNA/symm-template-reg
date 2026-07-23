import tempfile, unittest
from pathlib import Path
from tools.recheck_coordinate_guided_surface import _source_signatures

class RecheckIntegrityTest(unittest.TestCase):
    def test_signature_read_is_non_mutating(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'checkpoints').mkdir(); (root/'checkpoints/best.pth').write_bytes(b'x'); before=_source_signatures(root); after=_source_signatures(root)
            self.assertEqual(before,after)
