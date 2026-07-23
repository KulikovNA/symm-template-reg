import unittest
import torch
from symm_template_reg.geometry import barycentric_points, closest_points_on_triangle_mesh


class GTPatchCapacityTest(unittest.TestCase):
    def test_exact_barycentric_reconstruction(self):
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        points = torch.tensor([[.2, .3, 0.], [.7, .1, 0.]])
        nearest = closest_points_on_triangle_mesh(points, vertices, torch.tensor([[0, 1, 2]]))
        reconstructed = barycentric_points(vertices[torch.tensor([[0, 1, 2]])][nearest["face_ids"]], nearest["barycentric"])
        self.assertLess(float(torch.linalg.vector_norm(reconstructed - points, dim=-1).max()), 5e-4)

