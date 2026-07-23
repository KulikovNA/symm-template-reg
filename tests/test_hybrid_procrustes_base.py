import unittest

import torch

from symm_template_reg.models import build_model, register_all_modules
from tests.conditioned_test_utils import conditioned_batch, tiny_conditioned_config


class HybridProcrustesBaseTest(unittest.TestCase):
    def test_identity_initialized_correction(self):
        register_all_modules()
        config = tiny_conditioned_config(num_hypotheses=1, svd=True)
        config["base_pose_source"] = "procrustes_plus_direct_residual"
        config["sample_context_aggregator"]["split_rotation_translation"] = True
        config["base_pose_head"].update(split_rotation_translation=True, output_mode="bounded_correction")
        model = build_model(config).eval()
        with torch.no_grad(): prediction = model(conditioned_batch())
        self.assertTrue(torch.allclose(prediction.base_pose, prediction.correspondence_pose, atol=1e-6))


if __name__ == "__main__": unittest.main()
