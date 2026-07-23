from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from symm_template_reg.config import load_config
from symm_template_reg.engine.overfit_trainer import _build_dataset
from symm_template_reg.models import register_all_modules
from symm_template_reg.models.symmetry.groups import CyclicGroup
from symm_template_reg.models.symmetry.pose_conditioned_resolver import (
    PoseConditionedSymmetryResolver,
)
from symm_template_reg.models.symmetry.targets import build_fragment_symmetry_targets
from symm_template_reg.visualization.prediction_debug import _visible_points_gallery
from tests.test_fragment_symmetry_targets import DATASET_ROOT


@unittest.skipUnless(DATASET_ROOT.is_dir(), "real faces840 sample is unavailable")
class RealSampleC2ResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_all_modules()
        cls.temporary = tempfile.TemporaryDirectory()
        config = load_config("configs/debug/test_overfit_faces840_gpu.py")
        config["dataset"]["fragment_mesh_cache_dir"] = cls.temporary.name
        cls.config = config
        cls.dataset = _build_dataset(config)
        cls.index = next(
            index
            for index, record in enumerate(cls.dataset.sample_records)
            if record.scene_id == "scene_000005"
            and record.frame_id == 4
            and record.fragment_id == 0
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_real_visible_points_gt_and_resolver_both_produce_c2(self) -> None:
        sample = self.dataset[self.index]
        points_O = sample["gt"]["points_O_corresponding"]
        gt_target = build_fragment_symmetry_targets(
            points_O,
            sample["template"]["symmetry_metadata"],
            base_pose=sample["gt"]["T_C_from_O"],
            min_points=self.config["data"]["symmetry_region_activity"]["min_points"],
            min_fraction=self.config["data"]["symmetry_region_activity"]["min_fraction"],
            assignment_tolerance_m=self.config["data"]["symmetry_region_activity"]["boundary_tolerance_m"],
        )
        self.assertEqual(gt_target.active_regions.tolist(), [True, True, True, True])
        self.assertEqual(gt_target.effective_group, CyclicGroup(2))
        points_C = sample["observed"]["points_C"]
        result = PoseConditionedSymmetryResolver().resolve(
            points_C[None],
            torch.ones((1, len(points_C)), dtype=torch.bool),
            sample["gt"]["T_C_from_O"][None, None],
            [sample["template"]["symmetry_metadata"]],
            self.config["data"]["symmetry_region_activity"],
        )
        self.assertEqual(result.active_regions_per_pose[0][0].tolist(), [True] * 4)
        self.assertEqual(result.effective_group_per_pose[0][0], CyclicGroup(2))
        self.assertEqual(len(result.expanded_poses_per_base_pose[0][0]), 2)
        with tempfile.TemporaryDirectory() as temporary:
            gallery = _visible_points_gallery(
                Path(temporary) / "gallery.ply",
                sample,
                result.expanded_poses_per_base_pose[0][0],
                points_O,
                result.point_region_ids_per_pose[0][0],
                columns=2,
                spacing_scale=1.5,
                comments=("integration_test=true",),
            )
            self.assertEqual(gallery["template_copy_count"], 2)


if __name__ == "__main__":
    unittest.main()
