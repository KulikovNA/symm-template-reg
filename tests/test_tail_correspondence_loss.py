import unittest
import torch
from symm_template_reg.models.losses import top_tail_mean


class TailLossTest(unittest.TestCase):
    def test_ten_percent_bad_points_dominate_tail(self):
        errors=torch.cat((torch.zeros(90),torch.full((10,),.01)))
        self.assertAlmostEqual(float(errors.mean()),.001,places=7)
        self.assertAlmostEqual(float(top_tail_mean(errors,.1)),.01,places=7)

