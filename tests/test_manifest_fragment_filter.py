from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from symm_template_reg.datasets import FragmentTemplateRegistrationDataset
from symm_template_reg.engine.manifest import load_and_validate_manifest
from tests.dataset_test_utils import build_dataset


def write_manifest(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
    )


class ManifestFragmentFilterTest(unittest.TestCase):
    def test_threshold_mismatch_fails_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = build_dataset(base / "test")
            fragment_filter = {"enabled": True, "min_num_faces": 4}
            dataset = FragmentTemplateRegistrationDataset(
                root,
                min_observed_points=0,
                observed_policy="all_points",
                fragment_mesh_filter=fragment_filter,
                fragment_mesh_cache_dir=base / "cache",
                template_fine_points=4,
                template_coarse_points=2,
            )
            record = dataset.sample_records[0]
            metadata = record.fragment_mesh_metadata
            manifest = base / "manifest.json"
            write_manifest(
                manifest,
                {
                    "debug_training_on_test_split": True,
                    "results_are_not_final_evaluation": True,
                    "fragment_filter": fragment_filter,
                    "samples": [
                        {
                            "sample_id": record.sample_id,
                            "fragment_mesh_sha256": metadata.sha256,
                            "fragment_num_faces": metadata.num_faces,
                        }
                    ],
                },
            )
            config = {
                "debug_training_on_test_split": True,
                "results_are_not_final_evaluation": True,
                "data": {"fragment_mesh_filter": fragment_filter},
            }
            payload, digest = load_and_validate_manifest(manifest, config, dataset)
            self.assertEqual(len(payload["samples"]), 1)
            self.assertEqual(len(digest), 64)

            mismatched = {
                **config,
                "data": {
                    "fragment_mesh_filter": {"enabled": True, "min_num_faces": 5}
                },
            }
            with self.assertRaisesRegex(ValueError, "threshold does not match"):
                load_and_validate_manifest(manifest, mismatched, dataset)

    def test_changed_manifest_fails_sha_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = build_dataset(base / "test")
            fragment_filter = {"enabled": True, "min_num_faces": 4}
            dataset = FragmentTemplateRegistrationDataset(
                root,
                min_observed_points=0,
                fragment_mesh_filter=fragment_filter,
                fragment_mesh_cache_dir=base / "cache",
                template_fine_points=4,
                template_coarse_points=2,
            )
            manifest = base / "manifest.json"
            write_manifest(
                manifest,
                {
                    "debug_training_on_test_split": True,
                    "results_are_not_final_evaluation": True,
                    "fragment_filter": fragment_filter,
                    "samples": [],
                },
            )
            manifest.write_text(manifest.read_text() + " ", encoding="utf-8")
            config = {
                "debug_training_on_test_split": True,
                "results_are_not_final_evaluation": True,
                "data": {"fragment_mesh_filter": fragment_filter},
            }
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_and_validate_manifest(manifest, config, dataset)


if __name__ == "__main__":
    unittest.main()
