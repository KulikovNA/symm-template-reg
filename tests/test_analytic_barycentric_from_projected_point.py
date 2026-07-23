import unittest, torch
from symm_template_reg.models.geometry.triangle_targets import closest_barycentric_on_triangles


class AnalyticBarycentricTest(unittest.TestCase):
    def test_barycentric_reconstructs_projected_point(self):
        tri = torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]); q = torch.tensor([[.2,.3,.4]])
        out = closest_barycentric_on_triangles(q, tri); reconstructed = (out["barycentric"][...,None] * tri).sum(1)
        self.assertTrue(torch.allclose(reconstructed, out["points"], atol=1e-6)); self.assertAlmostEqual(float(out["barycentric"].sum()), 1, places=6)


if __name__ == "__main__": unittest.main()

