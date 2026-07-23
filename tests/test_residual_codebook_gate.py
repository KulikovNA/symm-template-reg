import unittest

from symm_template_reg.evaluation.readiness import residual_codebook_gate


class ResidualCodebookGateTest(unittest.TestCase):
    def test_static_codebook_fails(self):
        metrics = {"eval/residual_query_static_fraction": 0.8, "eval/query_static_codebook_score": 0.9}
        self.assertFalse(residual_codebook_gate(metrics)["passed"])
        good = {"eval/residual_query_static_fraction": 0.1, "eval/query_static_codebook_score": 0.2}
        self.assertTrue(residual_codebook_gate(good)["passed"])


if __name__ == "__main__": unittest.main()
