from __future__ import annotations

import unittest

import torch

from symm_template_reg.models import build_model, register_all_modules
from tests.conditioned_test_utils import conditioned_batch, tiny_conditioned_config


class CorrespondencePoseChangesWithInputTest(unittest.TestCase):
    def test_four_observed_clouds_produce_nonconstant_correspondence_pose(self) -> None:
        register_all_modules(); torch.manual_seed(9)
        config = tiny_conditioned_config(num_hypotheses=1, svd=True)
        config["base_pose_source"] = "weighted_procrustes"
        model = build_model(config).eval()
        batch = conditioned_batch()
        base = batch["observed"]["points_C"][:1]
        clouds = torch.cat((base, base + torch.tensor([0.02, 0.0, 0.0]), base + torch.tensor([0.0, 0.03, 0.0]), base + torch.tensor([0.01, -0.02, 0.04])))
        batch["observed"] = {"points_C": clouds, "valid_mask": torch.ones(4, 6, dtype=torch.bool)}
        batch["template"] = {"points_O": batch["template"]["points_O"][:1].expand(4, -1, -1).clone(), "valid_mask": torch.ones(4, 6, dtype=torch.bool)}
        batch["meta"] = [{"symmetry_available": False}] * 4
        with torch.no_grad(): poses = model(batch).correspondence_pose
        self.assertTrue(torch.isfinite(poses).all())
        self.assertGreater(float(torch.pdist(poses.flatten(1)).mean()), 1e-5)


if __name__ == "__main__":
    unittest.main()
