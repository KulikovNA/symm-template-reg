import unittest
import torch


class SoftCapacityTest(unittest.TestCase):
    def test_logits_can_select_exact_anchor_targets(self):
        anchors=torch.eye(3)
        logits=torch.nn.Parameter(torch.zeros((3,3)))
        optimizer=torch.optim.Adam([logits],lr=.2)
        for _ in range(100):
            optimizer.zero_grad();prediction=torch.softmax(logits/.1,-1)@anchors
            loss=(prediction-anchors).square().mean();loss.backward();optimizer.step()
        self.assertLess(float(torch.linalg.vector_norm(prediction.detach()-anchors,dim=-1).max()),1e-3)
