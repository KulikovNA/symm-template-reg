from __future__ import annotations

import unittest

import torch

from symm_template_reg.models import build_model, register_all_modules
from tests.conditioned_test_utils import conditioned_batch, tiny_conditioned_config


class ProcrustesBaseHasNoDirectRotationTest(unittest.TestCase):
    def test_base_head_forward_is_not_called(self) -> None:
        register_all_modules()
        config = tiny_conditioned_config(num_hypotheses=1, svd=True)
        config["base_pose_source"] = "weighted_procrustes"
        model = build_model(config).eval(); calls = []
        hook = model.base_pose_head.register_forward_hook(lambda *_: calls.append(1))
        with torch.no_grad(): prediction = model(conditioned_batch())
        hook.remove()
        self.assertEqual(calls, [])
        self.assertTrue(torch.equal(prediction.base_pose, prediction.correspondence_pose))


if __name__ == "__main__":
    unittest.main()
