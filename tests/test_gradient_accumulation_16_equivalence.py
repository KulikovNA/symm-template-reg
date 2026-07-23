import unittest, torch
from tests.optimization_test_utils import linear_accumulation

class GradientAccumulation16Test(unittest.TestCase):
    def test_batch8_accum2(self):
        a,_=linear_accumulation(16); b,_=linear_accumulation(8)
        for left,right in zip(a,b): self.assertTrue(torch.allclose(left,right,atol=1e-6,rtol=1e-6))

