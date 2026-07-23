from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from symm_template_reg.datasets import FragmentTemplateRegistrationDataset
from symm_template_reg.datasets.collate import packed_collate
from symm_template_reg.datasets.fragment_mesh_filter import (
    FragmentMeshFilter,
    FragmentMeshMetadata,
)
from tests.dataset_test_utils import build_dataset


def metadata(num_faces: int) -> FragmentMeshMetadata:
    return FragmentMeshMetadata(
        scene_id="scene_000000",
        fragment_id=0,
        fragment_key="scene_000000:fragment_0000",
        mesh_path=Path("fragment.ply"),
        num_vertices=10,
        num_faces=num_faces,
        surface_area_m2=0.01,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 1.0),
        bbox_diagonal_m=3.0**0.5,
        sha256="0" * 64,
        polygon_size_distribution={"3": num_faces},
        file_size=1,
        mtime_ns=1,
        annotation_path=Path("fragment_annotations.json"),
        annotation_sha256="1" * 64,
    )


class FragmentMeshFilterTest(unittest.TestCase):
    def test_face_threshold_boundary_and_no_implicit_upper_bound(self) -> None:
        filter_module = FragmentMeshFilter(
            {"enabled": True, "min_num_faces": 4, "max_num_faces": None}
        )
        self.assertTrue(filter_module.evaluate(metadata(4)).accepted)
        rejected = filter_module.evaluate(metadata(3))
        self.assertFalse(rejected.accepted)
        self.assertEqual(
            rejected.reasons, ["physical_fragment_num_faces_below_min"]
        )
        self.assertTrue(filter_module.evaluate(metadata(100000)).accepted)

    def test_enabled_filter_without_threshold_fails_clearly(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Fragment face threshold is enabled but min_num_faces is not configured",
        ):
            FragmentMeshFilter({"enabled": True, "min_num_faces": None})

    def test_dataset_excludes_every_observation_of_rejected_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = build_dataset(Path(temporary) / "test")
            dataset = FragmentTemplateRegistrationDataset(
                root,
                min_observed_points=0,
                observed_policy="all_points",
                fragment_mesh_filter={"enabled": True, "min_num_faces": 4},
                fragment_mesh_cache_dir=Path(temporary) / "cache",
                template_fine_points=4,
                template_coarse_points=2,
            )
            self.assertEqual(len(dataset), 2)
            for index in range(len(dataset)):
                self.assertTrue(
                    dataset[index]["meta"]["fragment_mesh"][
                        "passed_training_size_filter"
                    ]
                )
                self.assertGreaterEqual(
                    dataset[index]["meta"]["fragment_mesh"]["num_faces"], 4
                )
            batch = packed_collate([dataset[0], dataset[1]])
            self.assertEqual(len(batch["sample_id"]), 2)
            self.assertEqual(batch["observed"].batch_size, 2)

            with self.assertRaisesRegex(ValueError, "no usable samples"):
                FragmentTemplateRegistrationDataset(
                    root,
                    min_observed_points=0,
                    observed_policy="all_points",
                    fragment_mesh_filter={"enabled": True, "min_num_faces": 5},
                    fragment_mesh_cache_dir=Path(temporary) / "cache2",
                    template_fine_points=4,
                    template_coarse_points=2,
                )

    def test_physical_and_observed_rejections_have_distinct_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = build_dataset(Path(temporary) / "test")
            dataset = FragmentTemplateRegistrationDataset(
                root,
                min_observed_points=4,
                observed_policy="all_points",
                fragment_mesh_filter={"enabled": True, "min_num_faces": 4},
                fragment_mesh_cache_dir=Path(temporary) / "cache",
                template_fine_points=4,
                template_coarse_points=2,
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(
                dataset.index_report["rejected_observed_points_too_few"], 1
            )
            self.assertEqual(
                dataset.index_report["rejected_because_physical_fragment"], 0
            )
            self.assertEqual(
                dataset.index_report["rejected_samples"][0]["rejection_reason"],
                "observed_points_too_few",
            )


if __name__ == "__main__":
    unittest.main()
