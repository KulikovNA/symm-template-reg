import unittest, torch
from symm_template_reg.models.geometry.aux_guided_triangle_candidates import AuxGuidedTriangleCandidateBuilder

class NotCentroidOnlyTest(unittest.TestCase):
    def test_long_triangle_counterexample(self):
        v=torch.tensor([[0.,0,0],[100,0,0],[0,1,0],[1.,-.1,0],[1.2,-.1,0],[1.,.1,0]]); f=torch.tensor([[0,1,2],[3,4,5]]); q=torch.tensor([[[0.,0,0.]]])
        centroid=torch.cdist(q[0],v[f].mean(1)).argmin().item()
        exact=AuxGuidedTriangleCandidateBuilder(candidate_k=1)(q,[v],[f],torch.ones(1,1,dtype=torch.bool))['candidate_triangle_ids'].item()
        self.assertEqual(centroid,1); self.assertEqual(exact,0)
