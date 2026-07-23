import unittest, torch

class SizeBucketEquivalenceTest(unittest.TestCase):
    def test_per_sample_mean_is_permutation_invariant(self):
        torch.manual_seed(5); values=[torch.randn(n).square().mean() for n in (3,8,5,11)]
        self.assertTrue(torch.allclose(torch.stack(values).mean(),torch.stack([values[i] for i in (0,2,1,3)]).mean()))

