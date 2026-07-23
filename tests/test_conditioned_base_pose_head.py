from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.heads.conditioned_base_pose_head import (
    ConditionedBasePoseHead,
)


class ConditionedBasePoseHeadTest(unittest.TestCase):
    def test_base_pose_changes_with_sample_context_and_has_no_query(self) -> None:
        torch.manual_seed(2)
        head = ConditionedBasePoseHead(embed_dim=8, hidden_dim=16)
        contexts = torch.stack((torch.arange(8.0), torch.arange(8.0).flip(0)))
        result = head(contexts, torch.zeros(2, 3), torch.ones(2))
        self.assertEqual(result["base_T_C_from_O"].shape, (2, 4, 4))
        self.assertFalse(
            torch.allclose(
                result["base_pose_parameters_normalized"][0],
                result["base_pose_parameters_normalized"][1],
            )
        )
        self.assertFalse(hasattr(head, "query_embedding"))


if __name__ == "__main__":
    unittest.main()
