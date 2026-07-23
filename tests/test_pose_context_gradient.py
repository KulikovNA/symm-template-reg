from __future__ import annotations

import unittest

import torch

from symm_template_reg.models import build_model
from tests.conditioned_test_utils import conditioned_batch, tiny_conditioned_config


class PoseContextGradientTest(unittest.TestCase):
    def _observed_gradient(self, model: torch.nn.Module) -> float:
        return sum(
            float(parameter.grad.abs().sum())
            for parameter in model.observed_encoder.parameters()
            if parameter.grad is not None
        )

    def test_base_and_residual_gradients_reach_observed_encoder(self) -> None:
        torch.manual_seed(5)
        model = build_model(tiny_conditioned_config(num_hypotheses=3)).train()
        output = model(conditioned_batch())
        output.base_pose_parameters_normalized.square().mean().backward()
        self.assertGreater(self._observed_gradient(model), 0.0)
        model.zero_grad(set_to_none=True)
        output = model(conditioned_batch())
        output.residual_pose_parameters[..., 6:9].sum().backward()
        self.assertGreater(self._observed_gradient(model), 0.0)


if __name__ == "__main__":
    unittest.main()
