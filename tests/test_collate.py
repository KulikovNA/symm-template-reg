from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from symm_template_reg.datasets import (
    FragmentTemplateCollator,
    FragmentTemplateRegistrationDataset,
    PackedPointBatch,
    packed_collate,
    padded_collate,
)
from symm_template_reg.registry import COLLATE_FUNCTIONS, build_from_cfg

from tests.dataset_test_utils import build_dataset


class PackedPointBatchTest(unittest.TestCase):
    def test_pack_pad_split_roundtrip(self) -> None:
        first = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        second = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        packed = PackedPointBatch.from_list([first, second])
        self.assertEqual(packed.lengths.tolist(), [2, 5])
        self.assertEqual(packed.offsets.tolist(), [0, 2, 7])
        self.assertEqual(packed.batch_indices.tolist(), [0, 0, 1, 1, 1, 1, 1])
        packed.validate()
        dense = packed.to_padded()
        self.assertEqual(tuple(dense["points"].shape), (2, 5, 3))
        self.assertEqual(dense["valid_mask"].sum(1).tolist(), [2, 5])
        restored = PackedPointBatch.from_padded(dense)
        torch.testing.assert_close(restored.points, packed.points)
        self.assertEqual(restored.lengths.tolist(), [2, 5])
        split = restored.split()
        torch.testing.assert_close(split[0], first)
        torch.testing.assert_close(split[1], second)
        self.assertEqual(restored.to("cpu").device.type, "cpu")
        half = restored.to(dtype=torch.float16)
        self.assertEqual(half.points.dtype, torch.float16)
        self.assertEqual(half.lengths.dtype, torch.long)


class DatasetCollateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = build_dataset(Path(self.temporary.name) / "test")
        self.dataset = FragmentTemplateRegistrationDataset(
            root,
            min_observed_points=0,
            observed_policy="all_points",
            template_fine_points=4,
            template_coarse_points=2,
        )
        self.samples = [self.dataset[0], self.dataset[1]]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_packed_collate_with_different_lengths(self) -> None:
        batch = packed_collate(self.samples)
        self.assertEqual(batch["collate_mode"], "packed")
        self.assertEqual(batch["observed"].lengths.tolist(), [3, 5])
        self.assertEqual(tuple(batch["observed"].points.shape), (8, 3))
        self.assertEqual(batch["template"].lengths.tolist(), [4, 4])
        self.assertEqual(batch["gt"]["points_O_corresponding"].lengths.tolist(), [3, 5])
        self.assertEqual(tuple(batch["gt"]["T_C_from_O"].shape), (2, 4, 4))
        self.assertEqual(tuple(batch["gt"]["overlap_labels"].shape), (8,))
        self.assertEqual(batch["template_symmetry_metadata"], [None, None])

    def test_padded_valid_mask_and_label_padding(self) -> None:
        batch = padded_collate(self.samples)
        observed = batch["observed"]
        self.assertEqual(batch["collate_mode"], "padded")
        self.assertEqual(tuple(observed["points_C"].shape), (2, 5, 3))
        self.assertEqual(observed["valid_mask"].sum(1).tolist(), [3, 5])
        self.assertTrue(torch.equal(observed["surface_labels"][0, 3:], torch.tensor([255, 255])))
        self.assertEqual(tuple(batch["gt"]["points_O_corresponding"].shape), (2, 5, 3))
        self.assertFalse(batch["gt"]["overlap_labels"][0, 3:].any())

    def test_explicit_invalid_points_reach_padded_mask(self) -> None:
        self.samples[0]["observed"]["valid_mask"][1] = False
        packed = packed_collate(self.samples)
        self.assertFalse(packed["observed"].to_padded()["valid_mask"][0, 1])
        padded = padded_collate(self.samples)
        self.assertEqual(padded["observed"]["valid_mask"].sum(1).tolist(), [2, 5])
        self.assertEqual(padded["observed"]["lengths"].tolist(), [2, 5])

    def test_mixed_optional_targets_keep_per_sample_validity(self) -> None:
        self.samples[0]["gt"]["active_symmetry_regions"] = torch.tensor([True, False])
        self.samples[1]["gt"]["active_symmetry_regions"] = None
        self.samples[1]["gt"]["points_O_corresponding"] = None
        self.samples[1]["gt"]["overlap_labels"] = None
        self.samples[0]["observed"]["normals_C"] = torch.ones(3, 3)
        batch = packed_collate(self.samples)
        self.assertEqual(batch["gt"]["symmetry_supervision_mask"].tolist(), [True, False])
        self.assertEqual(
            batch["gt"]["active_symmetry_regions_valid_mask"].tolist(),
            [[True, True], [False, False]],
        )
        self.assertEqual(batch["gt"]["points_O_corresponding_valid_mask"].tolist(), [True] * 3 + [False] * 5)
        self.assertEqual(batch["gt"]["overlap_labels_valid_mask"].tolist(), [True] * 3 + [False] * 5)
        self.assertIn("normals_C", batch["observed"].features)
        self.assertEqual(
            batch["observed"].features["normals_C_valid_mask"].tolist(),
            [True] * 3 + [False] * 5,
        )
        padded = padded_collate(self.samples)
        self.assertEqual(
            padded["observed"]["normals_C_valid_mask"].sum(1).tolist(),
            [3, 0],
        )

    def test_downsampled_template_faces_are_not_misindexed(self) -> None:
        self.samples[0]["template"]["fine_points_O"] = self.samples[0]["template"]["points_O"][:2]
        self.samples[0]["template"]["fine_normals_O"] = self.samples[0]["template"]["normals_O"][:2]
        self.samples[0]["template"]["fine_indices"] = torch.tensor([0, 1])
        batch = packed_collate(self.samples)
        self.assertIsNone(batch["template_faces"][0])
        self.assertIsNotNone(batch["template_meshes"][0]["faces"])

    def test_collator_builds_from_registry_config(self) -> None:
        collator = build_from_cfg(
            {"type": "FragmentTemplateCollator", "mode": "packed"},
            COLLATE_FUNCTIONS,
        )
        self.assertIsInstance(collator, FragmentTemplateCollator)
        self.assertEqual(collator(self.samples)["collate_mode"], "packed")


if __name__ == "__main__":
    unittest.main()
