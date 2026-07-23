import unittest
import torch

from tools.audit_fine_feature_capacity import LinearTriangleDiagnostic


class FrozenFeatureTriangleCapacityTest(unittest.TestCase):
    def test_diagnostic_consumes_frozen_point_and_candidate_features(self):
        model = LinearTriangleDiagnostic(4)
        point = torch.randn(3,4); candidate = torch.randn(3,5,4)
        self.assertEqual(model(point, candidate).shape, (3,5))


if __name__ == "__main__": unittest.main()
