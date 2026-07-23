from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from symm_template_reg.visualization.prediction_debug import (
    _visible_points_gallery,
    export_prediction_visualizations,
)
from tests.test_fragment_symmetry_targets import metadata


class PredictionGeometrySourceTest(unittest.TestCase):
    def test_visible_gallery_repeats_only_the_supplied_observed_points(self) -> None:
        sample = {
            "template": {
                "points_O": torch.tensor(
                    [[0.0, -0.05, 0.0], [0.01, 0.05, 0.0], [-0.01, 0.05, 0.0]]
                ),
                "faces": torch.tensor([[0, 1, 2]]),
                "symmetry_metadata": metadata(),
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gallery.ply"
            result = _visible_points_gallery(
                path,
                sample,
                torch.eye(4).repeat(2, 1, 1),
                torch.tensor([[0.0, -0.02, 0.0], [0.0, 0.03, 0.0]]),
                torch.tensor([1, 3]),
                columns=2,
                spacing_scale=1.5,
                comments=("geometry_source=observed_visible_points_C",),
            )
            self.assertEqual(result["template_copy_count"], 2)
            self.assertEqual(result["observed_visible_points_copy_count"], 2)
            self.assertEqual(result["points_per_copy"], 2)
            self.assertFalse(result["uses_full_fragment_mesh"])
            header = path.read_text(encoding="ascii").split("end_header", 1)[0]
            self.assertIn("geometry_source=observed_visible_points_C", header)

    def test_main_export_uses_resolver_and_static_reference_root(self) -> None:
        source = inspect.getsource(export_prediction_visualizations)
        self.assertIn("PoseConditionedSymmetryResolver", source)
        self.assertIn('destination.parent / "reference"', source)
        self.assertNotIn('sample_dir / "gt_fragment_regions_on_template.ply"', source)
        self.assertIn('"full_fragment_mesh_used_for_main_group": False', source)


if __name__ == "__main__":
    unittest.main()
