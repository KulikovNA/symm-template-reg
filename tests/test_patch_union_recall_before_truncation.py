import unittest, torch
from symm_template_reg.models.geometry.aux_guided_triangle_candidates import AuxGuidedTriangleCandidateBuilder

class PatchUnionRecallTest(unittest.TestCase):
    def test_union_contains_all_faces_owned_by_predicted_patch(self):
        q=torch.zeros(1,1,3); v=torch.randn(6,3); f=torch.tensor([[0,1,2],[3,4,5]])
        out=AuxGuidedTriangleCandidateBuilder(mode='predicted_patch_union')(q,[v],[f],torch.ones(1,1,dtype=torch.bool),torch.tensor([[[1]]]),[torch.tensor([0,1])])
        self.assertEqual(out['candidate_triangle_ids'][0,0,0].item(),1)
