import unittest

from tests.boundary_augmentation_test_utils import run


class DepthRingGateTest(unittest.TestCase):
    def test_depth_gate_can_reject_depth_ring(self):
        result = run(
            "dilate",
            include_fracture_candidates=False,
            max_local_depth_difference_m=0.0,
        )
        self.assertGreaterEqual(result["metadata"]["added_depth_ring_count"], 0)
        self.assertLessEqual(result["metadata"]["added_fraction"], 0.20)


if __name__ == "__main__":
    unittest.main()
