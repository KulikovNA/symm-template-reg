import unittest

import torch

from symm_template_reg.models import build_model, register_all_modules
from tests.conditioned_test_utils import conditioned_batch, tiny_conditioned_config


class ProcrustesBaseSourceTest(unittest.TestCase):
    def test_base_is_correspondence_pose(self):
        register_all_modules()
        config = tiny_conditioned_config(num_hypotheses=1, svd=True)
        config["base_pose_source"] = "weighted_procrustes"
        model = build_model(config).eval()
        with torch.no_grad(): prediction = model(conditioned_batch())
        self.assertTrue(torch.allclose(prediction.base_pose, prediction.correspondence_pose))


if __name__ == "__main__": unittest.main()
