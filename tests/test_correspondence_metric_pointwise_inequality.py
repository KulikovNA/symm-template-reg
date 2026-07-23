import unittest
import torch
from symm_template_reg.geometry import closest_points_on_triangle_mesh


class MetricInequalityTest(unittest.TestCase):
    def test_pointwise_triangle_inequality(self):
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        faces = torch.tensor([[0, 1, 2]])
        gt = torch.tensor([[.2, .3, .01], [.8, .1, -.02]])
        pred = torch.tensor([[.2, .3, .4], [.8, .1, -.3]])
        d_pred = closest_points_on_triangle_mesh(pred, vertices, faces)["distances"]
        d_gt = closest_points_on_triangle_mesh(gt, vertices, faces)["distances"]
        self.assertTrue(torch.all(d_pred <= torch.linalg.vector_norm(pred - gt, dim=-1) + d_gt + 1e-6))

