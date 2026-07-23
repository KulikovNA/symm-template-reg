from __future__ import annotations

import unittest

import torch

from symm_template_reg.models.losses import DirectCorrespondencePoseConsistencyLoss


class DirectCorrespondencePoseConsistencyTest(unittest.TestCase):
    def test_identical_is_small_and_translation_is_penalized(self) -> None:
        direct = torch.eye(4).unsqueeze(0)
        same = DirectCorrespondencePoseConsistencyLoss()(direct, direct)
        moved = direct.clone()
        moved[:, 0, 3] = 0.1
        different = DirectCorrespondencePoseConsistencyLoss()(direct, moved)
        self.assertLess(float(same), 1e-3)
        self.assertGreater(float(different), 0.9)


if __name__ == "__main__":
    unittest.main()
