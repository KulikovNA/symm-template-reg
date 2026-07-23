from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from symm_template_reg.engine.overfit_trainer import _write_evaluation
from symm_template_reg.engine.view_ladder import query_world_consistency


class QueryAssignmentLoggingTest(unittest.TestCase):
    def test_matrix_shape_and_switch_rate_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {
                    "sample_id": "a",
                    "frame_id": 4,
                    "oracle_query_index": 0,
                    "query_pose_costs": [0.1, 0.2],
                    "query_rotation_error_deg": [1.0, 3.0],
                    "query_translation_error_mm": [1.0, 3.0],
                },
                {
                    "sample_id": "b",
                    "frame_id": 8,
                    "oracle_query_index": 1,
                    "query_pose_costs": [0.3, 0.1],
                    "query_rotation_error_deg": [3.0, 1.0],
                    "query_translation_error_mm": [3.0, 1.0],
                },
            ]
            _write_evaluation(root, 0, {}, rows)
            rows[0]["oracle_query_index"] = 1
            _write_evaluation(root, 1, {}, rows)
            with (root / "evaluations/epoch_0001/query_assignment_matrix.csv").open() as stream:
                matrix = list(csv.DictReader(stream))
            self.assertEqual((len(matrix), len(matrix[0]) - 3), (2, 2))
            diagnostics = json.loads(
                (root / "query_assignment_diagnostics.json").read_text()
            )
            self.assertEqual(diagnostics["query_assignment_comparison_count"], 2)
            self.assertEqual(diagnostics["query_assignment_switch_rate"], 0.5)

    def test_world_axis_spread_treats_opposite_directions_as_equivalent(self) -> None:
        identity = torch.eye(4, dtype=torch.float64)
        flipped = identity.clone()
        flipped[:3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        rows = [
            {
                "query_T_W_from_O": [identity.tolist()],
                "symmetry_axis_O": [0.0, 0.0, 1.0],
            },
            {
                "query_T_W_from_O": [flipped.tolist()],
                "symmetry_axis_O": [0.0, 0.0, 1.0],
            },
        ]
        result = query_world_consistency(rows)
        self.assertAlmostEqual(result["0"]["world_axis_spread_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
