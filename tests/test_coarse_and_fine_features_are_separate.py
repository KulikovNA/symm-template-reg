import unittest
from symm_template_reg.models.heads.surface_constrained_correspondence_head_v2 import SurfaceConstrainedCorrespondenceHeadV2


class SeparateFeaturesTest(unittest.TestCase):
    def test_adapter_without_candidate_head_is_rejected(self):
        with self.assertRaises(ValueError):
            SurfaceConstrainedCorrespondenceHeadV2(embed_dim=8, fine_feature_adapter={"type": "FineLocalCorrespondenceFeatureAdapter", "embed_dim": 8})


if __name__ == "__main__": unittest.main()

