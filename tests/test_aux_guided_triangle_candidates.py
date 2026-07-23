import unittest, torch
from symm_template_reg.models.geometry.aux_guided_triangle_candidates import AuxGuidedTriangleCandidateBuilder

class AuxCandidatesTest(unittest.TestCase):
    def test_exact_nearest_triangle_is_first(self):
        v=torch.tensor([[0.,0,0],[1,0,0],[0,1,0],[5,0,0],[6,0,0],[5,1,0]]); f=torch.tensor([[0,1,2],[3,4,5]])
        out=AuxGuidedTriangleCandidateBuilder(candidate_k=1)(torch.tensor([[[.2,.2,.1]]]),[v],[f],torch.ones(1,1,dtype=torch.bool))
        self.assertEqual(out['candidate_triangle_ids'].item(),0); self.assertFalse(out['selection_uses_centroid_distance'])
