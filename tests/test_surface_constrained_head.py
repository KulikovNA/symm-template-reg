import unittest
import torch
from symm_template_reg.geometry import closest_points_on_triangle_mesh
from symm_template_reg.models.heads import SurfaceConstrainedCorrespondenceHeadV2


class SurfaceHeadTest(unittest.TestCase):
    def test_triangle_mode_outputs_points_on_mesh(self):
        torch.manual_seed(2);dimension=8
        vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        faces=torch.tensor([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
        head=SurfaceConstrainedCorrespondenceHeadV2(embed_dim=dimension,num_patches=4,local_candidates=4)
        output=head(torch.randn(1,6,dimension),torch.randn(1,4,dimension),vertices[None],torch.ones(1,6,dtype=torch.bool),torch.ones(1,4,dtype=torch.bool),template_mesh_vertices_O=[vertices],template_mesh_faces=[faces])
        distance=closest_points_on_triangle_mesh(output["points_O"][0],vertices,faces)["distances"]
        self.assertLess(float(distance.detach().max()),2e-6)
