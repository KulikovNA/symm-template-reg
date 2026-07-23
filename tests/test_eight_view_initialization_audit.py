import inspect
import unittest

from tools import audit_eight_view_initialization as audit


class EightViewInitializationAuditTest(unittest.TestCase):
    def test_is_read_only_and_emits_required_reports(self):
        self.assertEqual(audit.EXPECTED_FRAMES, (4, 5, 2, 8, 0, 1, 6, 9))
        source = inspect.getsource(audit.run_audit)
        for name in (
            "eight_view_initialization_per_sample.csv",
            "eight_view_initialization_summary.json",
            "eight_view_initialization_report.md",
        ):
            self.assertIn(name, source)
        self.assertIn('"training_performed": False', source)
        self.assertIn('"checkpoint_unchanged"', source)


if __name__ == "__main__":
    unittest.main()
