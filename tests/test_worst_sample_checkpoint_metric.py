import unittest

from symm_template_reg.evaluation.active_coordinate import ten_view_sample_score, worst_ten_view_sample_score
from tests.test_three_gate_semantics import row


class WorstSampleMetricTest(unittest.TestCase):
    def test_worst_frame_not_mean_selects_checkpoint(self):
        rows = [row(frame, correspondence=.2, alignment=.2, rotation=.05, translation=.02) for frame in range(10)]
        rows[6] = row(6, correspondence=2.0, alignment=2.0, rotation=.5, translation=.5)
        self.assertEqual(worst_ten_view_sample_score(rows), ten_view_sample_score(rows[6]))


if __name__ == "__main__": unittest.main()
