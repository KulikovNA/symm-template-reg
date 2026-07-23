import unittest
import torch
from tools.audit_barycentric_feature_capacity import _triangle_descriptor


class BarycentricCapacityTest(unittest.TestCase):
    def test_descriptor_is_frame_internal(self):
        triangle = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        self.assertEqual(_triangle_descriptor(triangle).shape, (1, 16))


if __name__ == "__main__": unittest.main()
