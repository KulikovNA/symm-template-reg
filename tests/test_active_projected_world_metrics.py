import unittest
import torch

from symm_template_reg.evaluation.active_coordinate import active_world_metrics
from symm_template_reg.models.symmetry.metadata import SymmetryAxis, SymmetryMetadata


class ActiveProjectedWorldMetricsTest(unittest.TestCase):
    def test_reads_exact_and_k16_world_poses(self):
        eye = torch.eye(4).tolist()
        rows = [{"exact_global_T_W_from_O": eye, "k16_T_W_from_O": eye} for _ in range(2)]
        metadata = SymmetryMetadata(
            version=1, object_model_id="x", coordinate_frame="O",
            axis=SymmetryAxis("z", (0., 0., 0.), (0., 0., 1.)), regions=(),
        )
        values = active_world_metrics(rows, metadata, {"type": "C", "order": 2})
        self.assertEqual(values["exact_global_world_translation_range_mm"], 0.0)
        self.assertIn("k16_world_axis_spread_deg", values)


if __name__ == "__main__": unittest.main()
