import unittest
import torch
from symm_template_reg.engine.single_fragment import apply_trainable_prefixes, build_selective_optimizer_parameter_groups


class SelectiveGroupsTest(unittest.TestCase):
    def test_new_and_pretrained_lrs_are_disjoint(self):
        model = torch.nn.ModuleDict({"fine": torch.nn.Linear(2, 2), "pretrained": torch.nn.Linear(2, 2), "frozen": torch.nn.Linear(2, 2)})
        apply_trainable_prefixes(model, ("fine", "pretrained"))
        groups = build_selective_optimizer_parameter_groups(model, default_lr=3e-4, prefix_learning_rates={"pretrained": 3e-5})
        self.assertEqual({g["lr"] for g in groups}, {3e-4, 3e-5})


if __name__ == "__main__": unittest.main()

