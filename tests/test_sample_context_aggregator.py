from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.heads.sample_context import (
    SampleConditionedContextAggregator,
)


class SampleContextAggregatorTest(unittest.TestCase):
    def test_masked_attention_ignores_padding_and_reports_diagnostics(self) -> None:
        torch.manual_seed(1)
        module = SampleConditionedContextAggregator(embed_dim=8)
        observed = torch.randn(2, 4, 8)
        template = torch.randn(2, 5, 8)
        observed_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        template_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
        first = module(
            observed,
            template,
            observed_mask,
            template_mask,
            torch.zeros(2, 3),
            torch.ones(2),
        )
        observed[~observed_mask] = 1e6
        template[~template_mask] = -1e6
        second = module(
            observed,
            template,
            observed_mask,
            template_mask,
            torch.zeros(2, 3),
            torch.ones(2),
        )
        self.assertTrue(torch.allclose(first["sample_context"], second["sample_context"]))
        self.assertEqual(first["sample_context"].shape, (2, 8))
        self.assertEqual(first["sample_context_norm"].shape, (2,))

    def test_mean_max_mode_builds(self) -> None:
        module = SampleConditionedContextAggregator(
            embed_dim=8, aggregation="masked_mean_max_pooling"
        )
        tokens = torch.randn(1, 3, 8)
        result = module(
            tokens,
            tokens,
            torch.ones(1, 3, dtype=torch.bool),
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 3),
            torch.ones(1),
        )
        self.assertEqual(result["sample_context"].shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
