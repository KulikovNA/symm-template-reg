import unittest, torch
from symm_template_reg.geometry.triangle_surface import closest_points_on_triangle_mesh

@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
class ProjectionDeviceTest(unittest.TestCase):
    def test_cpu_cuda_consistency(self):
        torch.manual_seed(3); v=torch.rand(12,3); f=torch.arange(12).reshape(-1,3); q=torch.rand(8,3)
        cpu=closest_points_on_triangle_mesh(q,v,f,point_chunk_size=4)['points']; gpu=closest_points_on_triangle_mesh(q.cuda(),v.cuda(),f.cuda(),point_chunk_size=4)['points'].cpu()
        self.assertTrue(torch.allclose(cpu,gpu,atol=2e-6))
