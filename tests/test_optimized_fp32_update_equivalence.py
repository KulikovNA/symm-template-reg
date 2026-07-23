import unittest, torch
from tests.optimization_test_utils import linear_accumulation

class OptimizedUpdateTest(unittest.TestCase):
    def test_one_step_update(self):
        _,a=linear_accumulation(16); _,b=linear_accumulation(16)
        for left,right in zip(a,b): self.assertTrue(torch.equal(left,right))

