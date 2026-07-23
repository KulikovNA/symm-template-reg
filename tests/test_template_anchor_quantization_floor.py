import unittest
import torch


class QuantizationFloorTest(unittest.TestCase):
    def test_nearest_anchor_floor_is_measurable(self):
        target=torch.tensor([[.5,0.,0.]])
        anchors=torch.tensor([[0.,0.,0.],[1.,0.,0.]])
        floor=torch.cdist(target[None],anchors[None]).amin()
        self.assertAlmostEqual(float(floor),.5,places=6)

