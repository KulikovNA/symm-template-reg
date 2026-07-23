import unittest, torch
from symm_template_reg.models.pose import WeightedProcrustes

class UniformWeightsTest(unittest.TestCase):
    def test_all_valid_points_have_equal_weight(self):
        q=torch.randn(1,9,3); m=torch.tensor([[1,1,1,1,1,1,0,0,0]],dtype=torch.bool); w=m.float()/m.sum(1,keepdim=True)
        out=WeightedProcrustes().solve(q,q,w,m)
        self.assertTrue(torch.allclose(out["normalized_weights"][0,m[0]],torch.full((6,),1/6)))
        self.assertEqual(float(out["effective_correspondence_count"]),6.0)
