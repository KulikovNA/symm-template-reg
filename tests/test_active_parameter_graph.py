import unittest

from tests.clean_v3_test_utils import tiny_gradient_snapshot


class ActiveParameterGraphTest(unittest.TestCase):
    def test_real_q_aux_backward_reaches_graph(self):
        names, missing = tiny_gradient_snapshot()
        self.assertGreater(len(names), 0)
        self.assertEqual(missing, ())


if __name__ == "__main__": unittest.main()
