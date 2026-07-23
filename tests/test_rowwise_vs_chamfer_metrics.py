import unittest
import torch
from symm_template_reg.evaluation.correspondence_diagnostics import rowwise_and_chamfer


class RowwiseChamferTest(unittest.TestCase):
    def test_permutation_only_breaks_rowwise_identity(self):
        target=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        result=rowwise_and_chamfer(target[[1,2,0]],target)
        self.assertGreater(float(result["rowwise_distance"].mean()),.5)
        self.assertLess(float(result["symmetric_chamfer_distance"]),1e-7)

