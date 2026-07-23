from __future__ import annotations

import unittest
from pathlib import Path

from symm_template_reg.config import load_config
from symm_template_reg.engine.single_fragment import apply_trainable_prefixes
from symm_template_reg.models import build_model, register_all_modules


class StageFreezingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules()
        cls.root = Path(__file__).resolve().parents[1]

    def freeze(self, filename):
        config = load_config(self.root / "configs/debug/single_fragment" / filename)
        model = build_model(config["model"])
        report = apply_trainable_prefixes(
            model, config["stage"]["trainable_module_prefixes"]
        )
        return model, report

    def test_stage01_excludes_logits_and_regions(self):
        model, _ = self.freeze("01_k8_pose_only.py")
        self.assertFalse(model.pose_head.logit_projection.weight.requires_grad)
        self.assertFalse(model.symmetry_head.point_classifier.weight.requires_grad)
        self.assertTrue(model.pose_head.pose_projection[-1].weight.requires_grad)

    def test_stage02_only_logit_projection(self):
        model, report = self.freeze("02_k8_ranking_only.py")
        names = report["trainable_parameter_names"]
        self.assertTrue(names)
        self.assertTrue(all(name.startswith("pose_head.logit_projection.") for name in names))

    def test_stage03_only_region_head(self):
        _, report = self.freeze("03_k8_regions_only.py")
        self.assertTrue(all(name.startswith("symmetry_head.") for name in report["trainable_parameter_names"]))

    def test_stage04_unfreezes_all(self):
        _, report = self.freeze("04_k8_joint_finetune.py")
        self.assertEqual(report["frozen_parameter_count"], 0)


if __name__ == "__main__":
    unittest.main()
