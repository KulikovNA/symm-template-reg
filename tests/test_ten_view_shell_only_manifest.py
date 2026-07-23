import unittest

from symm_template_reg.engine.single_fragment import validate_single_fragment_manifest_payload
from symm_template_reg.engine.view_ladder import subset_view_manifest
from tests.test_single_frame_manifest import source_manifest


class TenViewManifestTest(unittest.TestCase):
    def test_exact_ten_view_shell_only_contract(self):
        payload = subset_view_manifest(source_manifest(), tuple(range(10)))
        payload.update(initialization_mode="scratch", pretrained_checkpoint=None, registration_point_selection="shell_only")
        for sample in payload["samples"]:
            sample.update(registration_point_selection="shell_only", fracture_point_count=0, effective_symmetry_group={"type": "C", "order": 2})
        report = validate_single_fragment_manifest_payload(payload, expected_samples=10)
        self.assertEqual(report["frame_ids"], list(range(10)))
        self.assertEqual(payload["train_sample_ids"], payload["validation_sample_ids"])
        self.assertTrue(all(row["fracture_point_count"] == 0 for row in payload["samples"]))
        self.assertTrue(all(row["effective_symmetry_group"] == {"type": "C", "order": 2} for row in payload["samples"]))


if __name__ == "__main__": unittest.main()
