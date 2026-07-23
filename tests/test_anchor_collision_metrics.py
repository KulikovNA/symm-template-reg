import unittest
import torch
from symm_template_reg.evaluation.correspondence_diagnostics import attention_distribution_metrics


class AnchorCollisionTest(unittest.TestCase):
    def test_single_popular_anchor_is_reported(self):
        logits=torch.zeros((10,4));logits[:,0]=10
        metric=attention_distribution_metrics(logits)
        self.assertEqual(int(metric["unique_argmax_anchors"]),1)
        self.assertAlmostEqual(float(metric["collision_ratio"]),.9,places=6)
        self.assertEqual(float(metric["most_popular_anchor_fraction"]),1.)

