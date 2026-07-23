import unittest, torch
from symm_template_reg.models.pose.weighted_procrustes import WeightedProcrustes

class BatchedProcrustesTest(unittest.TestCase):
    def test_batch_matches_individual_solves(self):
        torch.manual_seed(4); source=torch.randn(3,30,3); rotation=torch.eye(3).expand(3,3,3); translation=torch.randn(3,3)*.01
        target=source@rotation.transpose(-1,-2)+translation[:,None]; mask=torch.ones(3,30,dtype=torch.bool); weights=mask.float()
        solver=WeightedProcrustes(); batch=solver.solve(source,target,weights,mask)["transform"]
        rows=torch.cat([solver.solve(source[i:i+1],target[i:i+1],weights[i:i+1],mask[i:i+1])["transform"] for i in range(3)])
        self.assertTrue(torch.allclose(batch,rows,atol=2e-6,rtol=1e-6))

