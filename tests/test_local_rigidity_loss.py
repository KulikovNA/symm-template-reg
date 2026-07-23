import unittest
import torch
from symm_template_reg.evaluation.correspondence_diagnostics import local_rigidity_errors


class LocalRigidityTest(unittest.TestCase):
    def test_collapse_is_penalized(self):
        observed=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        rigid=local_rigidity_errors(observed,observed,3).mean()
        collapsed=local_rigidity_errors(torch.zeros_like(observed),observed,3).mean()
        self.assertLess(float(rigid),1e-7)
        self.assertGreater(float(collapsed),.5)

