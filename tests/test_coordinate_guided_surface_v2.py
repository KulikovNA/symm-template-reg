import unittest, torch
from symm_template_reg.models.heads.coordinate_guided_surface_correspondence_v2 import CoordinateGuidedSurfaceCorrespondenceV2

class SurfaceV2Test(unittest.TestCase):
    def test_surface_and_pose_outputs_without_learned_heads(self):
        v=torch.tensor([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]); f=torch.tensor([[0,1,2],[0,1,3],[0,2,3],[1,2,3]]); q=torch.tensor([[[.2,.2,0],[.2,0,.2],[0,.2,.2],[.3,.3,.4]]]); mask=torch.ones(1,4,dtype=torch.bool)
        out=CoordinateGuidedSurfaceCorrespondenceV2(candidate_k=2,projection_mode='shortlist',fallback_to_global_exact=False)(q,q,[v],[f],mask)
        self.assertFalse(out['learned_barycentric_head_used']); self.assertEqual(out['T_C_from_O'].shape,(1,4,4))
