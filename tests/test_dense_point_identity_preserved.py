import unittest
import torch
from symm_template_reg.models.geometry.fine_local_correspondence_features import FineLocalCorrespondenceFeatureAdapter


class DenseIdentityTest(unittest.TestCase):
    def test_distinct_dense_rows_are_not_broadcast(self):
        n, d = 40, 8; adapter = FineLocalCorrespondenceFeatureAdapter(embed_dim=d)
        points = torch.arange(n).float()[None, :, None].repeat(1, 1, 3) * 1e-3
        mask = torch.ones(1, n, dtype=torch.bool); raw = torch.zeros(1, n, d); raw[0, :, 0] = torch.arange(n)
        result = adapter(raw, torch.zeros_like(raw), points, mask, torch.ones_like(points))["fine_point_features"]
        self.assertGreater(torch.unique(result[0], dim=0).shape[0], 1)


if __name__ == "__main__": unittest.main()

