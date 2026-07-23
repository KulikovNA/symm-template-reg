import unittest, torch
class TestAccumulation(unittest.TestCase):
    def test_equal_microbatches_match_batch_mean(self):
        values=torch.arange(16,dtype=torch.float32); expected=values.mean()
        for width in (16,8,4,2): self.assertTrue(torch.allclose(torch.stack([chunk.mean() for chunk in values.split(width)]).mean(),expected))

