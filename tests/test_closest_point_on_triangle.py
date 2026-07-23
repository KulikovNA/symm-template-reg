import unittest, torch
from symm_template_reg.models.geometry.triangle_targets import closest_barycentric_on_triangles


class ClosestTrianglePointTest(unittest.TestCase):
    def test_projection_hits_triangle_interior(self):
        triangle = torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]); q = torch.tensor([[.2,.3,2.]])
        result = closest_barycentric_on_triangles(q, triangle)
        self.assertTrue(torch.allclose(result["points"], torch.tensor([[.2,.3,0.]]), atol=1e-6))


if __name__ == "__main__": unittest.main()

