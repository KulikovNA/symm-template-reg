import unittest

import torch

from symm_template_reg.evaluation.active_coordinate import active_world_metrics, active_world_pairwise_matrices
from tests.test_fragment_symmetry_targets import metadata


class TenViewWorldMetricsTest(unittest.TestCase):
    def test_identical_ten_world_poses_have_zero_spread(self):
        eye = torch.eye(4).tolist()
        rows = [{"exact_global_T_W_from_O": eye, "k16_T_W_from_O": eye} for _ in range(10)]
        metrics = active_world_metrics(rows, metadata(), {"type": "C", "order": 2})
        matrices = active_world_pairwise_matrices(rows, metadata(), {"type": "C", "order": 2})
        self.assertAlmostEqual(metrics["exact_global_world_translation_range_mm"], 0.0)
        for matrix in matrices.values():
            self.assertEqual((len(matrix), len(matrix[0])), (10, 10))
            self.assertAlmostEqual(max(max(row) for row in matrix), 0.0)


if __name__ == "__main__": unittest.main()
