import unittest

from tests.boundary_augmentation_test_utils import run


class FractureGateTest(unittest.TestCase):
    def test_template_distance_rejects_fracture_candidates(self):
        result = run("dilate", max_pseudo_target_distance_m=1e-9)
        metadata = result["metadata"]
        self.assertGreater(metadata["fracture_candidates_total"], 0)
        self.assertGreater(
            metadata["fracture_candidates_rejected_by_template_distance"], 0
        )


if __name__ == "__main__":
    unittest.main()
