import unittest

from symm_template_reg.evaluation.patch_stage import top1_quality_gate


class Top1QualityGateTest(unittest.TestCase):
    def test_diagnostic_gate_does_not_block_stage_b(self):
        gate = top1_quality_gate({"valid_patch_set_top1_accuracy": 0.94})
        self.assertFalse(gate["top1_quality_passed"])
        self.assertFalse(gate["blocks_teacher_forced_local_stage_b"])


if __name__ == "__main__":
    unittest.main()
