import unittest
import torch


class NormalizedCoordinateTest(unittest.TestCase):
    def test_bbox_roundtrip(self):
        lo, hi = torch.tensor([-1., 2., 4.]), torch.tensor([3., 6., 8.]); q = torch.tensor([[0., 3., 7.]])
        n = 2 * (q - lo) / (hi - lo) - 1; decoded = .5 * (n + 1) * (hi - lo) + lo
        self.assertTrue(torch.allclose(q, decoded))


if __name__ == "__main__": unittest.main()

