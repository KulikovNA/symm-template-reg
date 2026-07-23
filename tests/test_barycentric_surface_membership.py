import unittest
import torch
from symm_template_reg.geometry import barycentric_points


class BarycentricMembershipTest(unittest.TestCase):
    def test_weights_are_nonnegative_and_normalized(self):
        triangle=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
        point=barycentric_points(triangle,torch.tensor([[-1.,2.,3.]]))
        self.assertAlmostEqual(float(point[0,2]),0.)
        self.assertGreaterEqual(float(point[0,0]),0.)
        self.assertGreaterEqual(float(point[0,1]),0.)
        self.assertLessEqual(float(point[0,0]+point[0,1]),1.)

