import unittest

from symm_template_reg.evaluation.active_coordinate import strict_and_practical_stage_gates
from tests.test_strict_and_practical_gates_are_separate import row


class InactiveLegacyNaNTest(unittest.TestCase):
    def test_inactive_nan_is_outside_active_gate(self):
        frames = (4, 5, 2, 8, 0, 1, 6, 9)
        rows = []
        for frame in frames:
            active = row(frame, 0.5)
            active["exact_global_projection_alignment_p95_mm"] = 0.5
            active["eval/inactive/legacy_barycentric/loss"] = float("nan")
            active["eval/inactive/legacy_pose_query/error"] = float("inf")
            rows.append(active)
        gates = strict_and_practical_stage_gates(rows, frames)
        self.assertTrue(gates["strict_submillimetre_gate"]["stage_passed"])
        self.assertTrue(gates["practical_pose_first_gate"]["stage_passed"])


if __name__ == "__main__":
    unittest.main()
