from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from symm_template_reg.datasets.template_repository import load_ply
from symm_template_reg.visualization.symmetry_debug import (
    create_unique_run_directory,
    run_annotated_fragment_symmetry_debug,
)


DATASET_ROOT = Path(
    "/home/nikita/data_generator/generation_dataset/generation_synthetic/output/"
    "fragment_template_registration/differBig/2026-07-08/test"
)
TEMPLATE = DATASET_ROOT / "models" / "object_000004__scale_0p05.ply"


def ply_header_counts(path: Path) -> tuple[int, int, str]:
    header = path.read_bytes().split(b"end_header", 1)[0].decode("ascii")
    vertices = faces = -1
    for line in header.splitlines():
        fields = line.split()
        if fields[:2] == ["element", "vertex"]:
            vertices = int(fields[2])
        if fields[:2] == ["element", "face"]:
            faces = int(fields[2])
    return vertices, faces, header


class UniqueRunDirectoryTest(unittest.TestCase):
    def test_collision_suffix_is_three_digits_and_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = create_unique_run_directory(temporary, timestamp="fixed")
            marker = first / "marker"
            marker.write_text("preserve", encoding="utf-8")
            second = create_unique_run_directory(temporary, timestamp="fixed")
            self.assertEqual(second.name, "symmetry_debug_fixed_001")
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


@unittest.skipUnless(TEMPLATE.is_file(), "real annotated fragment fixture unavailable")
class RealHypothesisGalleryTest(unittest.TestCase):
    def test_gallery_colors_fragment_footprint_on_template_without_fragment_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = run_annotated_fragment_symmetry_debug(
                dataset_root=DATASET_ROOT,
                object_model_id="object_000004",
                scene_ids=("scene_000000",),
                fragment_ids=(1,),
                mode="all",
                output_root=temporary,
                timestamp="gallery",
            )
            fragment_dir = (
                run_dir
                / "scenes"
                / "scene_000000"
                / "fragments"
                / "fragment_0001"
            )
            index = json.loads(
                (fragment_dir / "hypothesis_index.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (fragment_dir / "fragment_summary.json").read_text(encoding="utf-8")
            )
            pose_count = len(index["hypotheses"])
            self.assertEqual(index["gallery_layout"]["template_copy_count"], pose_count)
            self.assertEqual(index["gallery_layout"]["fragment_copy_count"], 0)
            self.assertEqual(index["gallery_layout"]["fragments_per_template_copy"], 0)
            self.assertEqual(summary["gallery_template_copy_count"], pose_count)
            self.assertEqual(summary["gallery_fragment_copy_count"], 0)
            self.assertTrue(summary["projection_uses_shell_faces_only"])
            self.assertGreater(summary["identity_projected_template_faces"], 0)
            self.assertGreater(summary["identity_boundary_split_source_faces"], 0)
            self.assertLess(
                summary["identity_projected_area_relative_error_vs_fragment_shell"],
                0.02,
            )
            self.assertGreater(
                summary["identity_output_template_faces"], len(load_ply(TEMPLATE)["faces"])
            )
            self.assertTrue(summary["hypothesis_math_checks"]["identity_matches_original"])
            self.assertTrue(summary["hypothesis_math_checks"]["axis_and_origin_fixed"])

            template_mesh = load_ply(TEMPLATE)
            source_fragment = DATASET_ROOT / "scene_000000/fragments/fragment_0001.ply"
            copied_fragment = fragment_dir / "fragment_0001.ply"
            self.assertEqual(copied_fragment.read_bytes(), source_fragment.read_bytes())
            _, boundary_faces, _ = ply_header_counts(
                run_dir / "template" / "template_region_boundaries.ply"
            )
            _, projected_faces, _ = ply_header_counts(
                fragment_dir / "fragment_regions_on_template.ply"
            )
            # The projected-region artifact is the template, the regular
            # markers, and one 12-face reference tube -- never fragment faces.
            self.assertEqual(
                projected_faces,
                summary["identity_output_template_faces"] + boundary_faces + 12,
            )
            _, gallery_faces, header = ply_header_counts(fragment_dir / "hypothesis_gallery.ply")
            # Per cell: one complete template and five 6-sided tube markers
            # (axis, XYZ origin, reference X), 12 faces each. The fragment is
            # represented only by colors on template faces.
            expected_faces = sum(
                hypothesis["output_template_faces"] + 60
                for hypothesis in index["hypotheses"]
            )
            self.assertEqual(gallery_faces, expected_faces)
            self.assertIn("property uchar red", header)
            self.assertIn("property uchar green", header)
            self.assertIn("property uchar blue", header)

            run_summary = json.loads(
                (run_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(run_summary["frame_files_enumerated"])
            self.assertEqual(run_summary["forbidden_modalities_read"], [])


if __name__ == "__main__":
    unittest.main()
