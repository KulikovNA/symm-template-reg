from __future__ import annotations

import unittest
from pathlib import Path

from symm_template_reg.config import load_config, validate_data_policy


class ConfigPolicyValidationTest(unittest.TestCase):
    def test_conflicting_legacy_point_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting observed point policies"):
            validate_data_policy(
                {
                    "data": {
                        "observed_filter": {"point_policy": "farthest_point_up_to_max"}
                    },
                    "dataset": {"observed_policy": "all_points"},
                }
            )

    def test_v2_has_one_policy_and_excludes_validation_fragments(self) -> None:
        config = load_config("configs/debug/test_overfit_faces840_gpu.py")
        self.assertEqual(
            config["data"]["observed_filter"]["point_policy"],
            "farthest_point_up_to_max",
        )
        self.assertNotIn("observed_policy", config["dataset"])
        self.assertEqual(
            config["data"]["fragment_mesh_filter"]["validation_policy"],
            "exclude",
        )

    def test_legacy_config_is_archived(self) -> None:
        self.assertTrue(
            Path(
                "configs/debug/archive/"
                "test_overfit_faces840_gpu_pose_only_legacy_20260716.py"
            ).is_file()
        )


if __name__ == "__main__":
    unittest.main()
