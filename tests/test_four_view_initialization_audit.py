import unittest
from pathlib import Path


class FourViewInitializationAuditTest(unittest.TestCase):
    def test_tool_declares_no_training(self):
        text = Path("tools/audit_four_view_initialization.py").read_text()
        self.assertIn('"training_performed": False', text)
        self.assertIn("four_view_initialization_per_sample.csv", text)


if __name__ == "__main__": unittest.main()

