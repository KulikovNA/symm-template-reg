from __future__ import annotations

import copy
import unittest

import torch

from symm_template_reg.models import build_model, register_all_modules
from tests.conditioned_test_utils import conditioned_batch, tiny_conditioned_config


class TrainedCheckpointTargetLeakageTest(unittest.TestCase):
    def test_target_counterfactual_is_invariant_after_optimizer_update(self) -> None:
        register_all_modules()
        model = build_model(tiny_conditioned_config())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        batch = conditioned_batch()
        prediction = model(batch)
        prediction.correspondence_points_O.square().mean().backward()
        optimizer.step()
        batch["gt"] = {"points_O_corresponding": torch.randn(2, 6, 3), "T_C_from_O": torch.eye(4).repeat(2, 1, 1)}
        changed = copy.deepcopy(batch)
        changed["gt"]["points_O_corresponding"].add_(100.0)
        changed["gt"]["T_C_from_O"][:, :3, 3].add_(100.0)
        model.eval()
        with torch.no_grad():
            original_output = model(batch)
            changed_output = model(changed)
        self.assertTrue(torch.equal(original_output.correspondence_points_O, changed_output.correspondence_points_O))
        self.assertTrue(torch.equal(original_output.base_pose, changed_output.base_pose))


if __name__ == "__main__":
    unittest.main()
