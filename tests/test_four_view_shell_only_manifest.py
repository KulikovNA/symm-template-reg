import json
import unittest
from pathlib import Path


class FourViewShellOnlyManifestTest(unittest.TestCase):
    def test_built_manifest_contract_when_present(self):
        path = Path("/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/coordinate_guided_surface/fragment0002_views04_shell_only.json")
        if not path.is_file():
            self.skipTest("automatic manifest build has not run yet")
        value = json.loads(path.read_text())
        self.assertEqual([sample["frame_id"] for sample in value["samples"]], [4, 5, 2, 8])
        self.assertTrue(all(sample["fracture_point_count"] == 0 for sample in value["samples"]))


if __name__ == "__main__": unittest.main()

