from __future__ import annotations

import unittest
from pathlib import Path

from symm_template_reg.config import load_config


class K8PoseOnlyConfigTest(unittest.TestCase):
    def test_only_symmetry_pose_is_weighted(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/debug/single_fragment/01_k8_pose_only.py")
        loss = config["loss"]
        self.assertEqual(config["model"]["pose_head"]["num_queries"], 8)
        self.assertEqual(loss["symmetry_pose_weight"], 1.0)
        self.assertEqual(loss["pose_query_ranking"]["weight"], 0.0)
        for name in (
            "observed_region_weight", "active_region_weight", "region_consistency_weight",
            "correspondence_weight", "overlap_weight", "uncertainty_weight",
            "insufficient_information_weight",
        ):
            self.assertEqual(loss[name], 0.0)
        self.assertEqual(config["train"]["best_metric"], "eval/oracle_best_pose_cost")


if __name__ == "__main__":
    unittest.main()
