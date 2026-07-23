import unittest
from tests.joint_test_utils import fixture, call

class SurfaceLossTest(unittest.TestCase):
    def test_off_surface_is_positive(self):
        c,q,t,p,m,meta,g=fixture(); out=call(c,q+.01,t,q,t,p,m,meta,g)
        self.assertGreater(float(out["loss_template_surface_normalized"]),1)
