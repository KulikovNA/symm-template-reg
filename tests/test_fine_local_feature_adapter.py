import unittest
import torch

from symm_template_reg.models.geometry.fine_local_correspondence_features import FineLocalCorrespondenceFeatureAdapter


class FineLocalFeatureAdapterTest(unittest.TestCase):
    def test_shapes_and_frame_local_geometry(self):
        torch.manual_seed(1); n, d = 40, 16
        adapter = FineLocalCorrespondenceFeatureAdapter(embed_dim=d)
        points = torch.randn(1, n, 3) * .01; mask = torch.ones(1, n, dtype=torch.bool)
        normals = torch.nn.functional.normalize(torch.randn_like(points), dim=-1)
        result = adapter(torch.randn(1, n, d), torch.randn(1, n, d), points, mask, normals)
        self.assertEqual(result["fine_point_features"].shape, (1, n, d))
        self.assertEqual(result["observed_local_geometry"].shape[-1], 30)


if __name__ == "__main__": unittest.main()

