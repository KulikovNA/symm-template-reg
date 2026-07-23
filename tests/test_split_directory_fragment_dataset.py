import os
import tempfile
import unittest
from pathlib import Path

from symm_template_reg.datasets import SplitDirectoryFragmentDataset


class SplitDirectoryFragmentDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get("FRAG_DATASET_ROOT")
        if not value:
            raise unittest.SkipTest("FRAG_DATASET_ROOT is not set")
        cls.root = Path(value)

    def _dataset(self, split, cache):
        return SplitDirectoryFragmentDataset(
            self.root,
            split=split,
            selector={"scene_ids": ["scene_000000"], "max_samples": 2},
            index_cache_dir=cache,
            boundary_augmentation={"enabled": False},
        )

    def test_split_qualified_ids_and_sample_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._dataset("val", temporary)
            sample = dataset[0]
        self.assertTrue(sample["sample_id"].startswith("val/"))
        for key in (
            "points_C", "target_points_O", "valid_mask", "T_C_from_O",
            "fragment_mesh_sha256", "template_sha256",
            "symmetry_sidecar_sha256", "augmentation_metadata",
        ):
            self.assertIn(key, sample)
        self.assertEqual(sample["points_C"].shape, sample["target_points_O"].shape)

    def test_index_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = self._dataset("train", first)
            b = self._dataset("train", second)
        self.assertEqual(a.index_fingerprint, b.index_fingerprint)
        self.assertEqual(
            [record.sample_id for record in a.sample_records],
            [record.sample_id for record in b.sample_records],
        )

    def test_splits_are_physically_isolated(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            train = self._dataset("train", first)
            val = self._dataset("val", second)
        self.assertTrue(all("/train/" in str(row.visible_points_path) for row in train.sample_records))
        self.assertTrue(all("/val/" in str(row.visible_points_path) for row in val.sample_records))
        self.assertNotEqual(train.index_fingerprint, val.index_fingerprint)

    def test_non_train_augmentation_is_rejected(self):
        with self.assertRaises(ValueError):
            SplitDirectoryFragmentDataset(
                self.root,
                split="test",
                selector={"max_samples": 1},
                boundary_augmentation={"enabled": True},
            )


if __name__ == "__main__":
    unittest.main()
