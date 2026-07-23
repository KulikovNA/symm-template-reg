import unittest
import torch
from symm_template_reg.geometry import closest_points_on_triangle_mesh


class ExactSurfaceMetricTest(unittest.TestCase):
    def test_triangle_interior_is_not_vertex_distance(self):
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        point = torch.tensor([[.25, .25, 0.]])
        exact = closest_points_on_triangle_mesh(point, vertices, torch.tensor([[0, 1, 2]]))["distances"]
        self.assertLess(float(exact), 1e-7)
        self.assertGreater(float(torch.cdist(point, vertices).amin()), .3)

