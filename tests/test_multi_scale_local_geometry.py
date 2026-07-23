import unittest
from symm_template_reg.models.geometry.fine_local_correspondence_features import FineLocalCorrespondenceFeatureAdapter


class MultiScaleGeometryTest(unittest.TestCase):
    def test_contract_is_exactly_three_scales(self):
        self.assertEqual(FineLocalCorrespondenceFeatureAdapter(embed_dim=8).knn_scales, (8, 16, 32))
        with self.assertRaises(ValueError): FineLocalCorrespondenceFeatureAdapter(embed_dim=8, knn_scales=(8, 16))


if __name__ == "__main__": unittest.main()

