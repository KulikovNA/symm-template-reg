import unittest, torch
from symm_template_reg.models.heads.coordinate_guided_surface_correspondence_v2 import CoordinateGuidedSurfaceCorrespondenceV2

class FallbackReportingTest(unittest.TestCase):
    def test_fallback_fraction_is_visible(self):
        v=torch.tensor([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]); f=torch.tensor([[0,1,2],[0,1,3],[0,2,3],[1,2,3]]); q=torch.rand(1,4,3); mask=torch.ones(1,4,dtype=torch.bool)
        out=CoordinateGuidedSurfaceCorrespondenceV2(candidate_k=1,projection_mode='shortlist',fallback_to_global_exact=True)(q,q,[v],[f],mask,shortlist_pass_mask=torch.zeros_like(mask))
        self.assertEqual(float(out['shortlist_fallback_fraction']),1.)
