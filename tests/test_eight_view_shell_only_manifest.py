import unittest

from symm_template_reg.engine.view_ladder import subset_view_manifest
from tests.test_single_frame_manifest import source_manifest


class EightViewShellOnlyManifestTest(unittest.TestCase):
    def test_exact_order_and_one_physical_fragment(self):
        frames = [4, 5, 2, 8, 0, 1, 6, 9]
        payload = subset_view_manifest(source_manifest(), frames)
        payload["registration_point_selection"] = "shell_only"
        self.assertEqual([row["frame_id"] for row in payload["samples"]], frames)
        self.assertEqual({row["fragment_mesh_sha256"] for row in payload["samples"]}, {"mesh"})
        self.assertEqual(payload["train_sample_ids"], payload["validation_sample_ids"])
        self.assertEqual(payload["registration_point_selection"], "shell_only")


if __name__ == "__main__":
    unittest.main()
