import unittest

import torch

from symm_template_reg.models.geometry.patch_targets import valid_set_topk_hits


class ValidSetTopKMetricsTest(unittest.TestCase):
    def test_topk_uses_any_valid_class(self):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
        valid = torch.tensor(
            [[False, True, False, False], [False, False, False, True]]
        )
        self.assertEqual(valid_set_topk_hits(logits, valid, 1).tolist(), [False, False])
        self.assertEqual(valid_set_topk_hits(logits, valid, 2).tolist(), [True, False])
        self.assertEqual(valid_set_topk_hits(logits, valid, 4).tolist(), [True, True])


if __name__ == "__main__":
    unittest.main()
