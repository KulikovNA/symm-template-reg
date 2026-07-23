from __future__ import annotations

import unittest

import torch

from symm_template_reg.engine.view_ladder import direct_optimize_pose_parameters
from tests.test_fragment_symmetry_targets import metadata


class DirectPoseParameterOptimizationTest(unittest.TestCase):
    def test_sixteen_random_starts_converge_without_network(self) -> None:
        gt = torch.eye(4, dtype=torch.float64)
        gt[:3, 3] = torch.tensor([0.02, -0.01, 0.5], dtype=torch.float64)
        points = torch.tensor(
            [[0.01, 0.00, 0.49], [0.03, -0.02, 0.51], [0.02, 0.01, 0.50]],
            dtype=torch.float64,
        )
        rows = direct_optimize_pose_parameters(
            gt_pose=gt,
            observed_points_C=points,
            symmetry_metadata=metadata(),
            effective_group={"type": "C", "order": 2},
            num_starts=16,
            steps=1200,
            learning_rate=0.05,
            seed=9,
        )
        self.assertGreaterEqual(
            sum(bool(row["success_0p1deg_0p1mm"]) for row in rows), 15
        )


if __name__ == "__main__":
    unittest.main()
