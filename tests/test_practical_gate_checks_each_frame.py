import unittest

from symm_template_reg.evaluation.active_coordinate import strict_and_practical_stage_gates
from tests.test_strict_and_practical_gates_are_separate import row


class PracticalGateEachFrameTest(unittest.TestCase):
    def test_one_bad_frame_blocks_good_average(self):
        frames = (4, 5, 2, 8, 0, 1, 6, 9)
        rows = [row(frame, 1.4) for frame in frames]
        rows[-1] = row(9, 1.50001)
        result = strict_and_practical_stage_gates(rows, frames)
        self.assertFalse(result["practical_pose_first_gate"]["stage_passed"])
        self.assertIn("frame_9", result["practical_pose_first_gate"]["failures"])


if __name__ == "__main__":
    unittest.main()
