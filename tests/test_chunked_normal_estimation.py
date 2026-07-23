import unittest

import torch

from symm_template_reg.models.geometry.ppf import (
    _estimate_unoriented_normals,
    chunked_eigvalsh_3x3,
)


class ChunkedNormalEstimationTest(unittest.TestCase):
    def test_chunk_size_does_not_change_unoriented_normals(self):
        torch.manual_seed(4)
        points = torch.randn(3, 41, 3)
        mask = torch.ones((3, 41), dtype=torch.bool)
        small = _estimate_unoriented_normals(
            points, mask, 8, eigh_chunk_size=7
        )
        large = _estimate_unoriented_normals(
            points, mask, 8, eigh_chunk_size=10000
        )
        # Eigenvector signs are arbitrary, hence compare unoriented lines.
        agreement = (small * large).sum(-1).abs()
        self.assertTrue(torch.allclose(agreement, torch.ones_like(agreement), atol=1e-5))

    def test_invalid_chunk_size_is_rejected(self):
        with self.assertRaises(ValueError):
            _estimate_unoriented_normals(
                torch.randn(1, 4, 3), torch.ones((1, 4), dtype=torch.bool),
                3, eigh_chunk_size=0,
            )

    def test_chunked_eigenvalues_equal_direct_result(self):
        torch.manual_seed(8)
        matrix = torch.randn(2, 17, 3, 3)
        covariance = matrix.transpose(-1, -2) @ matrix
        expected = torch.linalg.eigvalsh(covariance)
        actual = chunked_eigvalsh_3x3(covariance, chunk_size=5)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
