import unittest

import torch

from symm_template_reg.models.heads.sample_context import SampleConditionedContextAggregator


class SplitContextTest(unittest.TestCase):
    def test_rotation_and_translation_outputs_exist(self):
        module = SampleConditionedContextAggregator(embed_dim=8, split_rotation_translation=True)
        tokens = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.bool)
        output = module(tokens, tokens, mask, mask, torch.randn(2, 3), torch.ones(2))
        self.assertEqual(output["rotation_context"].shape, (2, 8))
        self.assertEqual(output["translation_context"].shape, (2, 8))


if __name__ == "__main__": unittest.main()
