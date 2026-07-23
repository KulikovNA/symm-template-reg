import unittest
import torch
from symm_template_reg.evaluation.correspondence_diagnostics import attention_distribution_metrics


class AttentionEntropyTest(unittest.TestCase):
    def test_uniform_is_more_diffuse_than_sharp(self):
        uniform=attention_distribution_metrics(torch.zeros((4,8)))
        sharp_logits=torch.full((4,8),-20.);sharp_logits[:,0]=20
        sharp=attention_distribution_metrics(sharp_logits)
        self.assertGreater(float(uniform["entropy"].mean()),float(sharp["entropy"].mean()))

