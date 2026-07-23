import unittest
from pathlib import Path

from tools.check_forbidden_imports import scan


class ProductionImportTest(unittest.TestCase):
    def test_no_adjacent_or_legacy_runtime_imports(self):
        root = Path(__file__).resolve().parents[1] / "symm_template_reg"
        self.assertEqual(scan(root), [])


if __name__ == "__main__":
    unittest.main()
