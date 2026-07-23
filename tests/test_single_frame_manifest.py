from __future__ import annotations

import unittest

from symm_template_reg.engine.single_fragment import (
    manifest_content_sha256,
    validate_single_fragment_manifest_payload,
)
from symm_template_reg.engine.view_ladder import subset_view_manifest


def source_manifest() -> dict:
    samples = [
        {
            "sample_id": f"scene_000000/frame_{frame:06d}/fragment_0002",
            "scene_id": "scene_000000",
            "frame_id": frame,
            "fragment_id": 2,
            "fragment_mesh_sha256": "mesh",
            "fragment_num_faces": 2318,
            "T_W_from_C_available": True,
        }
        for frame in range(10)
    ]
    payload = {
        "manifest_type": "single_fragment_overfit",
        "scene_id": "scene_000000",
        "fragment_id": 2,
        "fragment_mesh_sha256": "mesh",
        "accepted_observations": 10,
        "train_sample_ids": [item["sample_id"] for item in samples],
        "validation_sample_ids": [item["sample_id"] for item in samples],
        "samples": samples,
    }
    payload["manifest_sha256"] = manifest_content_sha256(payload)
    return payload


class SingleFrameManifestTest(unittest.TestCase):
    def test_one_frame_contains_exactly_one_physical_sample(self) -> None:
        payload = subset_view_manifest(source_manifest(), [4])
        report = validate_single_fragment_manifest_payload(payload, expected_samples=1)
        self.assertEqual(report["frame_ids"], [4])
        self.assertEqual(payload["train_sample_ids"], payload["validation_sample_ids"])
        self.assertTrue(payload["view_ladder"]["deterministic_point_order"])


if __name__ == "__main__":
    unittest.main()
