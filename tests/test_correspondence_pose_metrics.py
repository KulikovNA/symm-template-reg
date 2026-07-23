import types
import unittest

import torch

from symm_template_reg.engine.metrics import batch_pose_metric_rows
from tests.test_fragment_symmetry_targets import metadata


class CorrespondencePoseMetricsTest(unittest.TestCase):
    def test_perfect_correspondence_metrics(self):
        pose = torch.eye(4)[None]
        points = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
        prediction = types.SimpleNamespace(
            pose_hypotheses=pose[:, None], pose_logits=torch.zeros(1, 1),
            active_region_logits=None, observed_region_logits=None,
            base_pose=pose, context_diagnostics=None, residual_transforms=None,
            correspondence_pose=pose, correspondence_points_O=points,
            correspondence_confidence=torch.ones(1, 2),
            observed_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        batch = {
            "sample_id": ["x"], "scene_id": ["s"], "fragment_id": torch.tensor([0]),
            "template_symmetry_metadata": [metadata()],
            "gt": {"T_C_from_O": pose, "effective_symmetry_group": [{"type": "C", "order": 1}],
                   "active_symmetry_regions": None, "active_symmetry_regions_valid_mask": None,
                   "points_O_corresponding": points},
            "meta": [{"fragment_mesh": {"num_faces": 900, "surface_area_m2": 1., "bbox_diagonal_m": 1.}}],
        }
        row = batch_pose_metric_rows(prediction, batch)[0]
        self.assertLess(row["correspondence_point_rmse_mm"], 1e-8)
        self.assertTrue(row["correspondence_pose_success_2deg_2mm"])


if __name__ == "__main__": unittest.main()
