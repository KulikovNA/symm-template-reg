import unittest
from symm_template_reg.models.heads.coordinate_guided_triangle_head import CoordinateGuidedTriangleHead


class GuidedTriangleInputsTest(unittest.TestCase):
    def test_required_inputs_are_explicit(self):
        self.assertEqual(set(CoordinateGuidedTriangleHead.required_pair_inputs), {'fine_observed_feature','candidate_triangle_feature','q_aux_O','closest_point_O','q_aux_to_triangle_distance','triangle_normal_O','triangle_edge_lengths','coarse_patch_feature'})


if __name__ == "__main__": unittest.main()
