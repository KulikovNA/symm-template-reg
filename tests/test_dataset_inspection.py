from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.inspect_dataset import inspect_dataset

from tests.dataset_test_utils import build_dataset


class DatasetInspectionTest(unittest.TestCase):
    def test_missing_core_file_becomes_report_instead_of_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = build_dataset(Path(temporary) / "test")
            (root / "scene_000000" / "gt_annotations.json").unlink()
            output = Path(temporary) / "inspection"
            summary = inspect_dataset(root, output)
            self.assertEqual(summary["num_samples"], 0)
            self.assertGreater(summary["error_findings"], 0)
            for name in (
                "dataset_inventory.json",
                "dataset_inventory.md",
                "npz_schema.json",
                "sample_index.csv",
                "template_inventory.json",
                "warnings.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            warnings = json.loads((output / "warnings.json").read_text())["warnings"]
            codes = {warning["code"] for warning in warnings}
            self.assertIn("missing_required_file", codes)
            self.assertIn("missing_expected_path", codes)


if __name__ == "__main__":
    unittest.main()
