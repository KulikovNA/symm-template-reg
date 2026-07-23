import json
import unittest
from pathlib import Path


class ProvenanceTest(unittest.TestCase):
    def test_all_ported_targets_have_exact_header_evidence(self):
        root = Path(__file__).resolve().parents[1]
        modules = json.loads((root / "third_party_modules.json").read_text(encoding="utf-8"))
        for module in modules["modules"]:
            if module["status"] not in {"ported", "tested"}:
                continue
            for relative_path in module.get("target_paths", []):
                path = root / relative_path
                self.assertTrue(path.is_file(), f"missing target {relative_path}")
                header = path.read_text(encoding="utf-8")[:5000]
                self.assertIn(module["source_commit"], header, relative_path)
                self.assertIn(module["source_license"], header, relative_path)
                for original in module["original_paths"]:
                    self.assertIn(original, header, f"{relative_path}: {original}")


if __name__ == "__main__":
    unittest.main()
