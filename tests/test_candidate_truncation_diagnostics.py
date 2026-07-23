import unittest, torch
from tools.audit_triangle_candidate_pipeline import _recall

class TruncationDiagnosticsTest(unittest.TestCase):
    def test_loss_is_visible_after_truncation(self):
        valid=torch.zeros(1,40,dtype=torch.bool); valid[0,35]=True
        before=torch.arange(40)[None]; after=before[:,:32]
        self.assertTrue(_recall(before,before.ge(0),valid)[0]); self.assertFalse(_recall(after,after.ge(0),valid)[0])
