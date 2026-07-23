import unittest
import torch

from symm_template_reg.models.geometry.triangle_targets import deduplicate_candidate_ids


class LocalCandidatesDeterministicTest(unittest.TestCase):
    def test_same_input_has_identical_order(self):
        value = torch.tensor([[7, 2, 7, 1, 2]])
        first = deduplicate_candidate_ids(value)[0]
        second = deduplicate_candidate_ids(value)[0]
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__": unittest.main()
