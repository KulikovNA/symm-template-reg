import unittest
import torch
from torch.nn import functional as F


class LocalGradientTest(unittest.TestCase):
    def test_selected_coarse_scores_receive_gradient(self):
        coarse = torch.randn(3, 4, requires_grad=True)
        topk = coarse.topk(2, -1).indices
        fine = torch.randn(3, 4, requires_grad=True) + coarse.gather(1, topk).repeat_interleave(2, -1)
        gradient = torch.autograd.grad(F.cross_entropy(fine, torch.tensor([0, 1, 2])), coarse)[0]
        self.assertGreater(float(gradient.norm()), 0.)

