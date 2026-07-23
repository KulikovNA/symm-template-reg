from __future__ import annotations

import unittest
import warnings

import torch

from symm_template_reg.models.heads.pose_query_head import (
    LegacyAbsolutePoseQueryHead,
    PoseQueryHead,
)
from symm_template_reg.models.heads.residual_pose_hypothesis_head import (
    ResidualPoseHypothesisHead,
)


class QueryRequiresContextTest(unittest.TestCase):
    def test_unconditioned_bypass_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unconditioned bypass"):
            ResidualPoseHypothesisHead(
                embed_dim=16,
                num_heads=4,
                query_conditioning=dict(
                    type="film",
                    apply_each_decoder_layer=True,
                    allow_unconditioned_bypass=True,
                ),
            )

    def test_forward_requires_context(self) -> None:
        head = ResidualPoseHypothesisHead(
            embed_dim=16, num_heads=4, num_hypotheses=2, num_decoder_layers=1
        )
        with self.assertRaisesRegex(ValueError, "sample_context"):
            head(
                None,
                torch.randn(1, 3, 16),
                torch.ones(1, 3, dtype=torch.bool),
                torch.eye(4).unsqueeze(0),
                torch.ones(1),
            )

    def test_legacy_alias_has_identical_state_contract_and_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = LegacyAbsolutePoseQueryHead(embed_dim=16, num_heads=4)
            alias = PoseQueryHead(embed_dim=16, num_heads=4)
        self.assertEqual(set(legacy.state_dict()), set(alias.state_dict()))
        self.assertTrue(any("pose codebook" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
