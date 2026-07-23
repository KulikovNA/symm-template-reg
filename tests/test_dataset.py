from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from symm_template_reg.datasets import FragmentTemplateRegistrationDataset

from tests.dataset_test_utils import build_dataset


class FragmentTemplateDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = build_dataset(Path(self.temporary.name) / "test")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_real_contract_shapes_and_transform_roundtrip(self) -> None:
        dataset = FragmentTemplateRegistrationDataset(
            self.root,
            min_observed_points=0,
            max_observed_points=16,
            observed_policy="all_points",
            template_fine_points=4,
            template_coarse_points=2,
        )
        self.assertEqual(len(dataset), 2)
        sample = dataset[1]
        self.assertEqual(sample["sample_id"], "scene_000000/frame_000000/fragment_0001")
        self.assertEqual(tuple(sample["observed"]["points_C"].shape), (5, 3))
        self.assertIsNone(sample["observed"]["normals_C"])
        self.assertEqual(sample["observed"]["surface_labels"].dtype, torch.long)
        self.assertEqual(sample["observed"]["valid_mask"].dtype, torch.bool)
        self.assertEqual(tuple(sample["template"]["points_O"].shape), (4, 3))
        self.assertEqual(tuple(sample["template"]["faces"].shape), (4, 3))
        self.assertEqual(tuple(sample["gt"]["T_C_from_O"].shape), (4, 4))
        points_O = sample["gt"]["points_O_corresponding"]
        transform = sample["gt"]["T_C_from_O"]
        reconstructed = points_O @ transform[:3, :3].T + transform[:3, 3]
        torch.testing.assert_close(reconstructed, sample["observed"]["points_C"])
        self.assertFalse(sample["meta"]["symmetry_available"])
        self.assertIsNone(sample["gt"]["active_symmetry_regions"])
        self.assertIsNone(sample["gt"]["effective_symmetry_group"])

    def test_cap_preserves_row_correspondence_and_variable_n(self) -> None:
        dataset = FragmentTemplateRegistrationDataset(
            self.root,
            min_observed_points=0,
            max_observed_points=4,
            observed_policy="farthest_point_up_to_max",
            template_fine_points=4,
            template_coarse_points=2,
        )
        self.assertEqual([len(dataset[i]["observed"]["points_C"]) for i in range(2)], [3, 4])
        sample = dataset[1]
        transform = sample["gt"]["T_C_from_O"]
        reconstructed = (
            sample["gt"]["points_O_corresponding"] @ transform[:3, :3].T
            + transform[:3, 3]
        )
        torch.testing.assert_close(reconstructed, sample["observed"]["points_C"])

    def test_common_root_plus_split(self) -> None:
        dataset = FragmentTemplateRegistrationDataset(
            self.root.parent,
            split="test",
            min_observed_points=0,
            observed_policy="all_points",
            template_fine_points=4,
            template_coarse_points=2,
        )
        self.assertEqual(len(dataset), 2)

    def test_voxel_policy_applies_below_maximum_too(self) -> None:
        dataset = FragmentTemplateRegistrationDataset(
            self.root,
            min_observed_points=0,
            max_observed_points=100,
            observed_policy="voxel_downsample",
            voxel_size_m=1.0,
            template_fine_points=4,
            template_coarse_points=2,
        )
        self.assertEqual(len(dataset[1]["observed"]["points_C"]), 1)

    def test_sidecar_enables_automatic_targets(self) -> None:
        sidecar_root = build_dataset(Path(self.temporary.name) / "with_symmetry", with_sidecar=True)
        dataset = FragmentTemplateRegistrationDataset(
            sidecar_root,
            min_observed_points=0,
            observed_policy="all_points",
            template_fine_points=4,
            template_coarse_points=2,
        )
        sample = dataset[0]
        self.assertTrue(sample["meta"]["symmetry_available"])
        self.assertEqual(sample["gt"]["active_symmetry_regions"].tolist(), [True])
        self.assertEqual(sample["gt"]["effective_symmetry_group"], {"order": 4, "type": "C"})
        self.assertEqual(tuple(sample["gt"]["equivalent_T_C_from_O"].shape), (4, 4, 4))

    def test_transform_cannot_mutate_shared_template_cache(self) -> None:
        def destructive_transform(sample):
            sample["template"]["points_O"].zero_()
            return sample

        dataset = FragmentTemplateRegistrationDataset(
            self.root,
            min_observed_points=0,
            observed_policy="all_points",
            template_fine_points=4,
            template_coarse_points=2,
            transform=destructive_transform,
        )
        _ = dataset[0]
        cached = dataset.template_repository.get("object_000004__scale_0p05")
        self.assertFalse(torch.equal(cached["points_O"], torch.zeros_like(cached["points_O"])))


if __name__ == "__main__":
    unittest.main()
