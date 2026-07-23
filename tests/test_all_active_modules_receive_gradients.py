import unittest

from tests.clean_v3_test_utils import tiny_gradient_snapshot


class ActiveModuleGradientTest(unittest.TestCase):
    def test_all_expected_top_level_modules_receive_gradients(self):
        names, missing = tiny_gradient_snapshot()
        self.assertEqual(missing, ())
        expected = {
            "observed_encoder", "template_encoder", "interaction_transformer",
            "dual_stream_geometry_encoder", "dense_observed_fine_projection",
            "fine_template_projection", "template_context_projection",
            "fine_feature_adapter", "canonical_coordinate_head",
        }
        present = {name.split(".", 1)[0] for name in names}
        self.assertTrue(expected <= present)


if __name__ == "__main__": unittest.main()
