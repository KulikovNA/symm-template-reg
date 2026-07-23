import unittest

import torch

from symm_template_reg.models import build_model
from tests.clean_v3_test_utils import synthetic_batch, tiny_clean_v3_config


class InputConditioningTest(unittest.TestCase):
    def test_q_aux_changes_with_observed_cloud(self):
        torch.manual_seed(5)
        model = build_model(tiny_clean_v3_config()).eval()
        with torch.no_grad():
            first = model(synthetic_batch(1, observed_offset=0.0)).correspondence_points_O
            second = model(synthetic_batch(1, observed_offset=0.02)).correspondence_points_O
        self.assertGreater(float((first - second).abs().max()), 1e-8)


if __name__ == "__main__": unittest.main()
