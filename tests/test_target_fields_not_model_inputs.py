import inspect
import unittest

from symm_template_reg.models.detectors.conditioned_symm_template_reg import ConditionedSymmTemplateReg


class TargetForwardContractTest(unittest.TestCase):
    def test_gt_fields_are_absent_from_forward_feature_contract(self):
        source = inspect.getsource(ConditionedSymmTemplateReg.forward)
        self.assertNotIn('batch["gt"]', source)
        for field in ("points_O_corresponding", "active_symmetry_regions", "effective_symmetry_group", "fragment_mesh"):
            self.assertNotIn(field, source)


if __name__ == "__main__": unittest.main()
