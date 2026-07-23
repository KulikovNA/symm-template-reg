import unittest, torch
from symm_template_reg.models.heads.coordinate_guided_surface_projection import CoordinateGuidedSurfaceProjectionHead


def fixture():
    vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[2.,0.,0.],[2.,1.,0.]])
    faces=torch.tensor([[0,1,2],[1,3,4]]); q=torch.tensor([[[.2,.2,.5],[1.8,.2,.3]]]); ids=torch.tensor([[[0,1],[0,1]]]); mask=torch.ones(1,2,dtype=torch.bool)
    return vertices,faces,q,ids,mask


class ProjectionHeadTest(unittest.TestCase):
    def test_output_lies_on_selected_triangle(self):
        v,f,q,ids,mask=fixture(); out=CoordinateGuidedSurfaceProjectionHead()(q,ids,[v],[f],mask)
        tri=v[f[out["selected_triangle_ids"][0]]]; reconstructed=(out["analytic_barycentric_coordinates"][0][...,None]*tri).sum(1)
        self.assertTrue(torch.allclose(reconstructed,out["surface_correspondence_points_O"][0],atol=1e-6))


if __name__ == "__main__": unittest.main()

