import unittest
import torch


class PatchRecallTest(unittest.TestCase):
    def test_topk_recall(self):
        logits = torch.tensor([[9., 8., 0., 0.], [0., 4., 3., 2.]])
        target = torch.tensor([0, 2])
        top = logits.topk(4, -1).indices
        self.assertEqual(float(top[:, :1].eq(target[:, None]).any(-1).float().mean()), .5)
        self.assertEqual(float(top[:, :4].eq(target[:, None]).any(-1).float().mean()), 1.)

