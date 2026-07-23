import unittest, torch
from tests.test_coordinate_guided_surface_projection import fixture
from symm_template_reg.models.heads.coordinate_guided_surface_projection import CoordinateGuidedSurfaceProjectionHead


class GlobalCandidateProjectionTest(unittest.TestCase):
    def test_restricted_wrong_candidate_differs(self):
        v,f,q,ids,mask=fixture(); head=CoordinateGuidedSurfaceProjectionHead()
        global_out=head(q[:,:1],ids[:,:1],[v],[f],mask[:,:1])["surface_correspondence_points_O"]
        wrong=head(q[:,:1],torch.ones(1,1,1,dtype=torch.long),[v],[f],mask[:,:1])["surface_correspondence_points_O"]
        self.assertFalse(torch.allclose(global_out,wrong))


if __name__ == "__main__": unittest.main()

