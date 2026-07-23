from __future__ import annotations

import inspect
import unittest

from symm_template_reg.models.detectors.conditioned_symm_template_reg import ConditionedSymmTemplateReg


class CorrespondenceTargetNotModelInputTest(unittest.TestCase):
    def test_forward_never_reads_correspondence_or_pose_targets(self) -> None:
        source = inspect.getsource(ConditionedSymmTemplateReg.forward)
        self.assertNotIn('batch["gt"]', source)
        self.assertNotIn("points_O_corresponding", source)
        self.assertNotIn("effective_symmetry_group", source)
        self.assertNotIn("fragment_mesh", source)


if __name__ == "__main__":
    unittest.main()
