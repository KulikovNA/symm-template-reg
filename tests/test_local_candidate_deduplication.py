import unittest
import torch

from symm_template_reg.models.geometry.triangle_targets import (
    deduplicate_candidate_ids,
    inject_valid_triangle_ids,
)


class LocalCandidateDeduplicationTest(unittest.TestCase):
    def test_stable_unique_ids_and_padding(self):
        ids, mask, _ = deduplicate_candidate_ids(torch.tensor([[4, 2, 4, 3, 2]]))
        self.assertEqual(ids[0, mask[0]].tolist(), [4, 2, 3])
        self.assertTrue(bool(ids[0, ~mask[0]].eq(-1).all()))

    def test_injection_never_evicts_an_existing_valid_triangle(self):
        candidates = torch.tensor([[10, 11, 12]])
        candidate_mask = torch.ones_like(candidates, dtype=torch.bool)
        valid = torch.zeros((1, 14), dtype=torch.bool)
        valid[0, 12] = True
        valid[0, 13] = True
        injected, injected_mask, _ = inject_valid_triangle_ids(
            candidates, candidate_mask, valid
        )
        present = set(injected[0, injected_mask[0]].tolist())
        self.assertTrue({12, 13}.issubset(present))
        self.assertEqual(len(present), 3)


if __name__ == "__main__": unittest.main()
