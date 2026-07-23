from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from symm_template_reg.datasets.template_repository import load_ply
from symm_template_reg.models.symmetry.metadata import (
    SymmetryMetadata,
    load_symmetry_metadata,
)
from symm_template_reg.models.symmetry.region_assignment import (
    assign_symmetry_regions,
    validate_region_partition,
)
from symm_template_reg.visualization.ply import write_colored_ply
from symm_template_reg.visualization.symmetry_debug import (
    _TriangleSurfaceIndex,
    _refine_template_at_fragment_boundary,
    _split_template_at_region_boundaries,
    apply_gallery_offset,
    create_unique_run_directory,
    gallery_offsets,
    run_symmetry_debug,
)


DATASET_ROOT = Path(
    "/home/nikita/data_generator/generation_dataset/generation_synthetic/output/"
    "fragment_template_registration/differBig/2026-07-08/test"
)
TEMPLATE = DATASET_ROOT / "models" / "object_000004__scale_0p05.ply"
SIDECAR = DATASET_ROOT / "models" / "object_000004__scale_0p05.symmetry.json"


@unittest.skipUnless(TEMPLATE.is_file() and SIDECAR.is_file(), "real symmetry fixture unavailable")
class RealTemplateSymmetryVisualizationTest(unittest.TestCase):
    def test_sidecar_bands_partition_every_template_vertex_and_face(self) -> None:
        mesh = load_ply(TEMPLATE)
        metadata = load_symmetry_metadata(
            SIDECAR, expected_object_model_id=TEMPLATE.stem
        )
        assert isinstance(metadata, SymmetryMetadata)
        self.assertEqual(
            metadata.region_ids, ("band_00", "band_01", "band_02", "band_03")
        )
        self.assertEqual(
            [region.rotation_group.name for region in metadata.regions],
            ["SO2", "C10", "SO2", "C4"],
        )
        for previous, following in zip(metadata.regions[:-1], metadata.regions[1:]):
            self.assertAlmostEqual(previous.y_max_m, following.y_min_m, places=12)
        result = validate_region_partition(
            torch.from_numpy(mesh["points"]),
            torch.from_numpy(mesh["faces"]),
            metadata,
        )
        self.assertTrue(result.coverage_ok)
        self.assertEqual(result.unassigned_vertices, 0)
        self.assertEqual(result.overlap_vertices, 0)
        self.assertEqual(result.unassigned_faces, 0)
        self.assertEqual(result.overlap_faces, 0)
        self.assertEqual(len(result.vertex_region_indices), len(mesh["points"]))
        self.assertEqual(len(result.face_region_indices), len(mesh["faces"]))

    def test_template_debug_run_has_rgb_and_never_reuses_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = run_symmetry_debug(
                dataset_root=DATASET_ROOT,
                object_model_id="object_000004",
                mode="template",
                output_root=temporary,
                timestamp="20260715_120000",
            )
            second = run_symmetry_debug(
                dataset_root=DATASET_ROOT,
                object_model_id="object_000004",
                mode="template",
                output_root=temporary,
                timestamp="20260715_120000",
            )
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())
            ply = first / "template" / "template_symmetry_regions_with_boundaries.ply"
            header = ply.read_bytes().split(b"end_header", 1)[0].decode("ascii")
            self.assertIn("property uchar red", header)
            self.assertIn("property uchar green", header)
            self.assertIn("property uchar blue", header)
            visual_mesh = load_ply(
                first / "template" / "template_symmetry_regions.ply"
            )
            visual_triangles = visual_mesh["points"][visual_mesh["faces"]]
            axial = visual_triangles[:, :, 1]
            real_metadata = load_symmetry_metadata(
                SIDECAR, expected_object_model_id=TEMPLATE.stem
            )
            for boundary in [
                region.y_max_m for region in real_metadata.regions[:-1]
            ]:
                crossing = (axial.min(axis=1) < boundary - 1e-7) & (
                    axial.max(axis=1) > boundary + 1e-7
                )
                self.assertFalse(bool(crossing.any()))
            summary = json.loads(
                (
                    first / "template" / "template_symmetry_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertGreater(
                summary["visual_template_faces"], summary["original_template_faces"]
            )
            self.assertAlmostEqual(
                summary["visual_surface_area_m2"],
                summary["original_surface_area_m2"],
                places=8,
            )


class SymmetryVisualizationPrimitiveTest(unittest.TestCase):
    def test_template_triangle_is_cut_exactly_at_axial_region_boundary(self) -> None:
        metadata = SymmetryMetadata.from_dict(
            {
                "version": 1,
                "object_model_id": "object_test",
                "coordinate_frame": "O",
                "axis": {"name": "y", "origin": [0, 0, 0], "direction": [0, 1, 0]},
                "regions": [
                    {
                        "region_id": "lower",
                        "y_min_m": -1.0,
                        "y_max_m": 0.0,
                        "rotation_group": {"type": "C", "order": 2},
                    },
                    {
                        "region_id": "upper",
                        "y_min_m": 0.0,
                        "y_max_m": 1.0,
                        "rotation_group": {"type": "C", "order": 4},
                    },
                ],
            }
        )
        points = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        split = _split_template_at_region_boundaries(
            points, np.asarray([[0, 1, 2]], dtype=np.int64), metadata
        )
        triangles = split.vertices[split.faces].astype(np.float64)
        areas = 0.5 * np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        self.assertEqual(split.split_source_faces, 1)
        self.assertAlmostEqual(float(areas.sum()), 2.0, places=6)
        self.assertAlmostEqual(float(areas[split.region_indices == 0].sum()), 0.5, places=6)
        self.assertAlmostEqual(float(areas[split.region_indices == 1].sum()), 1.5, places=6)
        axial = triangles[:, :, 1]
        self.assertFalse(bool(((axial.min(axis=1) < -1e-8) & (axial.max(axis=1) > 1e-8)).any()))

    def test_template_triangle_is_cut_at_fragment_shell_boundary(self) -> None:
        template_points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        template_faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        shell_triangles = np.asarray(
            [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]],
            dtype=np.float32,
        )
        refined = _refine_template_at_fragment_boundary(
            template_points=template_points,
            template_faces=template_faces,
            shell_surface=_TriangleSurfaceIndex(shell_triangles, 1e-3),
            template_to_fragment_surface=None,
            boundary_resolution_m=0.1,
            max_depth=1,
        )
        triangles = refined.vertices[refined.faces].astype(np.float64)
        areas = 0.5 * np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        self.assertEqual(refined.split_source_faces, 1)
        self.assertGreater(len(refined.faces), 1)
        self.assertAlmostEqual(float(areas.sum()), 0.5, places=6)
        self.assertAlmostEqual(
            float(areas[refined.projected_faces].sum()), 0.125, places=5
        )

    def test_internal_boundaries_are_half_open_and_last_upper_is_closed(self) -> None:
        metadata = SymmetryMetadata.from_dict(
            {
                "version": 1,
                "object_model_id": "object_test",
                "coordinate_frame": "O",
                "axis": {"name": "y", "origin": [0, 0, 0], "direction": [0, 1, 0]},
                "regions": [
                    {
                        "region_id": "lower",
                        "y_min_m": -1.0,
                        "y_max_m": 0.0,
                        "rotation_group": {"type": "C", "order": 2},
                    },
                    {
                        "region_id": "upper",
                        "y_min_m": 0.0,
                        "y_max_m": 1.0,
                        "rotation_group": {"type": "C", "order": 4},
                    },
                ],
            }
        )
        points = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        membership = assign_symmetry_regions(points, metadata)
        self.assertEqual(
            membership.tolist(), [[True, False], [False, True], [False, True]]
        )

    def test_ply_writer_emits_vertex_rgb_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "colored.ply"
            write_colored_ply(
                path,
                np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
                np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
            )
            header = path.read_text(encoding="ascii").split("end_header", 1)[0]
            self.assertIn("property uchar red", header)
            self.assertIn("property uchar green", header)
            self.assertIn("property uchar blue", header)
            with self.assertRaises(FileExistsError):
                write_colored_ply(path, np.zeros((1, 3)), np.zeros((1, 3), dtype=np.uint8))

    def test_gallery_offsets_preserve_local_relative_placement(self) -> None:
        template = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.2, -0.1]])
        fragment = torch.tensor([[0.1, -0.1, 0.3], [-0.2, 0.4, 0.2]])
        offsets = gallery_offsets(3, columns=2, spacing_m=2.0)
        for offset in offsets:
            shifted_template = apply_gallery_offset(template, offset)
            shifted_fragment = apply_gallery_offset(fragment, offset)
            torch.testing.assert_close(
                shifted_fragment - shifted_template, fragment - template
            )

    def test_unique_directory_suffix_prevents_timestamp_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = create_unique_run_directory(temporary, timestamp="same")
            marker = first / "marker.txt"
            marker.write_text("keep", encoding="utf-8")
            second = create_unique_run_directory(temporary, timestamp="same")
            self.assertNotEqual(first, second)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
