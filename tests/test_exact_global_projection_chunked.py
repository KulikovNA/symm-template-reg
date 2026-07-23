import unittest, torch
from symm_template_reg.geometry.triangle_surface import closest_points_on_triangle_mesh

class ChunkedProjectionTest(unittest.TestCase):
    def test_chunks_match_unchunked(self):
        torch.manual_seed(2); v=torch.rand(30,3); f=torch.arange(30).reshape(-1,3); q=torch.rand(17,3)
        a=closest_points_on_triangle_mesh(q,v,f,point_chunk_size=len(q)); b=closest_points_on_triangle_mesh(q,v,f,point_chunk_size=3)
        self.assertTrue(torch.allclose(a['points'],b['points'],atol=1e-7))
