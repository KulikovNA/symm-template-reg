from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from symm_template_reg.datasets.fragment_mesh_filter import scan_fragment_mesh_metadata
from tests.dataset_test_utils import build_dataset


class FragmentMeshMetadataCacheTest(unittest.TestCase):
    def test_cache_reuse_and_mesh_change_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = build_dataset(base / "test")
            cache = base / "cache"
            config = {"cache_metadata": True, "manifest_mismatch_policy": "error"}
            first, first_report = scan_fragment_mesh_metadata(
                root, filter_config=config, cache_dir=cache
            )
            self.assertEqual(len(first), 2)
            self.assertEqual(first_report["fragment_mesh_cache_hits"], 0)
            self.assertEqual(first_report["fragment_mesh_cache_misses"], 2)

            second, second_report = scan_fragment_mesh_metadata(
                root, filter_config=config, cache_dir=cache
            )
            self.assertEqual(second_report["fragment_mesh_cache_hits"], 2)
            self.assertEqual(second_report["fragment_mesh_cache_misses"], 0)

            mesh_path = root / "scene_000000/fragments/fragment_0000.ply"
            mesh_path.write_text(
                mesh_path.read_text(encoding="ascii") + "\n", encoding="ascii"
            )
            third, third_report = scan_fragment_mesh_metadata(
                root, filter_config=config, cache_dir=cache
            )
            self.assertEqual(third_report["fragment_mesh_cache_hits"], 1)
            self.assertEqual(third_report["fragment_mesh_cache_misses"], 1)
            self.assertNotEqual(first[("scene_000000", 0)].sha256, third[("scene_000000", 0)].sha256)

    def test_missing_mesh_policy_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = build_dataset(base / "test")
            (root / "scene_000000/fragments/fragment_0000.ply").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "physical fragment mesh"):
                scan_fragment_mesh_metadata(
                    root,
                    filter_config={"missing_mesh_policy": "error"},
                    cache_dir=base / "cache",
                )


if __name__ == "__main__":
    unittest.main()
