import unittest
from tests.joint_test_utils import fixture, call

class JointLossTest(unittest.TestCase):
    def test_perfect_is_near_zero_and_wrong_pose_is_not(self):
        c,q,t,p,m,meta,g=fixture(); good=call(c,q,t,q,t,p,m,meta,g)
        wrong=t.clone(); wrong[:,:3,3]+=0.01; bad=call(c,q,wrong,q,t,p,m,meta,g)
        self.assertLess(float(good["loss_total"]),1e-10)
        self.assertGreater(float(bad["loss_total"]),1.0)
        self.assertGreater(c.weights["rotation"],0)
