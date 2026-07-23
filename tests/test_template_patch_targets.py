import unittest
import torch
from symm_template_reg.geometry import closest_points_on_triangle_mesh


class PatchTargetTest(unittest.TestCase):
    def test_nearest_triangle_and_barycentric_reconstruct_target(self):
        vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        faces=torch.tensor([[0,1,2]]);target=torch.tensor([[.2,.4,0.]])
        nearest=closest_points_on_triangle_mesh(target,vertices,faces)
        reconstructed=(nearest["barycentric"][:,:,None]*vertices[faces[nearest["face_ids"]]]).sum(1)
        torch.testing.assert_close(reconstructed,target,atol=1e-6,rtol=0)

