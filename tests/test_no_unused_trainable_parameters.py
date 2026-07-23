import unittest

from tests.clean_v3_test_utils import tiny_gradient_snapshot


class NoUnusedTrainableTest(unittest.TestCase):
    def test_no_trainable_parameter_is_outside_q_aux_graph(self):
        _, missing = tiny_gradient_snapshot()
        self.assertEqual(missing, ())


if __name__ == "__main__": unittest.main()
