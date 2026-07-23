import unittest

import torch

from symm_template_reg.evaluation.active_coordinate import active_world_pairwise_matrices
from symm_template_reg.models.symmetry.metadata import SymmetryAxis, SymmetryMetadata


class EightViewWorldPoseMetricsTest(unittest.TestCase):
    def test_pairwise_matrices_are_eight_by_eight_and_zero(self):
        identity = torch.eye(4).tolist()
        rows = [{"exact_global_T_W_from_O": identity} for _ in range(8)]
        metadata = SymmetryMetadata(
            version=1, object_model_id="x", coordinate_frame="O",
            axis=SymmetryAxis("z", (0, 0, 0), (0, 0, 1)), regions=(),
        )
        result = active_world_pairwise_matrices(rows, metadata, {"type": "C", "order": 2})
        for matrix in result.values():
            self.assertEqual((len(matrix), len(matrix[0])), (8, 8))
            self.assertTrue(all(abs(value) < 1e-8 for line in matrix for value in line))


if __name__ == "__main__":
    unittest.main()
