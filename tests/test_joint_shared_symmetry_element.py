import unittest, torch
from tests.joint_test_utils import fixture, call
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms
from symm_template_reg.models.pose.pose_representation import transform_points

class SharedSymmetryTest(unittest.TestCase):
    def test_c2_equivalent_uses_one_element(self):
        c,q,t,p,m,meta,g=fixture(); s=symmetry_transforms(g,meta.axis.direction,meta.axis.origin,dtype=q.dtype)[1]
        qe=transform_points(torch.linalg.inv(s),q[0]).unsqueeze(0); te=(t[0]@s).unsqueeze(0); pe=transform_points(te,qe)
        out=call(c,qe,te,q,t,pe,m,meta,g,torch.cat((q,qe),1))
        self.assertEqual(int(out["selected_shared_symmetry_element"][0]),1)
        self.assertLess(float(out["loss_total"]),1e-9)
