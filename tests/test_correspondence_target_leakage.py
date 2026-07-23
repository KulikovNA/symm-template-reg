import copy
import unittest

import torch

from symm_template_reg.models import build_model, register_all_modules
from tests.conditioned_test_utils import conditioned_batch, tiny_conditioned_config


class CorrespondenceTargetLeakageTest(unittest.TestCase):
    def test_gt_counterfactual_does_not_change_outputs(self):
        register_all_modules()
        torch.manual_seed(0)
        model = build_model(tiny_conditioned_config()).eval()
        batch = conditioned_batch()
        batch["gt"] = {"points_O_corresponding": torch.randn(2, 6, 3), "T_C_from_O": torch.eye(4).repeat(2, 1, 1)}
        with torch.no_grad(): original = model(batch)
        changed = copy.deepcopy(batch)
        changed["gt"]["points_O_corresponding"].add_(100)
        changed["gt"]["T_C_from_O"][:, :3, 3].add_(100)
        with torch.no_grad(): counterfactual = model(changed)
        self.assertTrue(torch.equal(original.correspondence_points_O, counterfactual.correspondence_points_O))
        self.assertTrue(torch.equal(original.base_pose, counterfactual.base_pose))


if __name__ == "__main__": unittest.main()
