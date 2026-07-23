import unittest
import torch
from symm_template_reg.geometry import closest_points_on_triangle_mesh


class GTOnSurfaceTest(unittest.TestCase):
    def test_triangle_interior_has_zero_distance(self):
        vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        faces=torch.tensor([[0,1,2]])
        points=torch.tensor([[.2,.3,0.],[.1,.1,0.]])
        result=closest_points_on_triangle_mesh(points,vertices,faces)
        self.assertLess(float(torch.quantile(result["distances"],.95)),5e-7)

