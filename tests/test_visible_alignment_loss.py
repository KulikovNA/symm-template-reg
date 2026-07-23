import unittest
from tests.joint_test_utils import fixture, call

class AlignmentLossTest(unittest.TestCase):
    def test_misalignment_is_positive(self):
        c,q,t,p,m,meta,g=fixture(); wrong=t.clone(); wrong[:,:3,3]+=.01; out=call(c,q,wrong,q,t,p,m,meta,g)
        self.assertGreater(float(out["loss_alignment_normalized"]),1); self.assertGreater(c.weights["alignment"],0)
