from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.pose.pose_codec import PoseCodec
from symm_template_reg.models.pose.pose_representation import make_transform
from symm_template_reg.models.pose.rotation import axis_angle_to_matrix


class PoseCodecTest(unittest.TestCase):
    def test_float64_round_trip_below_1e_7_m(self) -> None:
        codec = PoseCodec()
        points = torch.tensor(
            [[[0.1, -0.2, 0.5], [0.3, 0.2, 0.8], [-0.1, 0.1, 0.6]]],
            dtype=torch.float64,
        )
        context = codec.context(points, torch.ones((1, 3), dtype=torch.bool))
        rotation = axis_angle_to_matrix(
            torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
        ).unsqueeze(0)
        transform = make_transform(
            rotation, torch.tensor([[0.12, -0.07, 0.61]], dtype=torch.float64)
        )
        encoded = codec.encode_transform(
            transform, context.observed_centroid_C, context.observed_scale
        )
        decoded = codec.decode_transform(
            encoded[..., :6],
            encoded[..., 6:],
            context.observed_centroid_C,
            context.observed_scale,
        )
        torch.testing.assert_close(decoded, transform, atol=1e-10, rtol=1e-10)
        self.assertLess(
            float(torch.linalg.vector_norm(decoded[:, :3, 3] - transform[:, :3, 3])),
            1e-7,
        )


if __name__ == "__main__":
    unittest.main()
